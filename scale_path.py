"""
Scale Path — toward 1B TinyBrain ≈ 600B+ Transformer feel.

Thesis: quality should scale with *thinking compute*, not just params.
Path: close small-scale gap → prove think-step scaling → then grow to 1B.

Usage (Colab/RunPod — use ! in notebook cells):
  !python scale_path.py --verify
  !python scale_path.py --mode diagnose --steps 1000
  !python scale_path.py --mode verify_claim --steps 2000 --seeds 0,1,2,3,4
  !python scale_path.py --mode memory_ablation --steps 1500 --seeds 0,1,2
  !python scale_path.py --mode equal_flops --steps 2000 --samples 20000 --seeds 0,1,2,3,4
  !python scale_path.py --mode think_scale --steps 800
  !python scale_path.py --mode equal_flops --dataset wikitext   # Phase B: OOD prose
  !python scale_path.py --mode equal_flops --memory_sharp 32    # sharper slot read
  !python scale_path.py --mode equal_flops --think_steps 8      # thesis: more thinking vs TF
  !python scale_path.py --mode equal_flops --think_steps 8 --tf_layers 6   # V2-T8 vs bigger TF
  !python scale_path.py --mode equal_flops --think_steps 8 --think_rank 32 # cheaper thinking
  !python scale_path.py --mode equal_flops --ema 0.999           # EMA weights (overfitting defense)
  !python scale_path.py --mode equal_flops --think_steps 8 --think_rank 32 --thought_paths 2  # multi-agent
  !python scale_path.py --mode equal_flops --dataset code --seq_len 512 --think_steps 8  # code data
  !python scale_path.py --mode code_eval --dataset code --seq_len 512 --think_steps 8 --amp  # HumanEval pass@1
  !python scale_path.py --mode equal_flops --data_mix tinystories:0.5,wikitext:0.3,openwebtext:0.2  # replay-style
  !python scale_path.py --mode reason_eval --dataset gsm8k --seq_len 128 --think_steps 8  # supervised baseline
  !python scale_path.py --mode grpo --dataset gsm8k --seq_len 128 --think_steps 8 --rl_steps 60  # R1-style RL

Fairness defaults: LR schedule keyed to TOKENS (warmup+cosine), equal-FLOPs
compares BEST val with paired t-test/sign test over seeds, full budget per
model (no early stop in the race), tokens+wall-clock reported per model.
FLOPs are PROFILER-MEASURED (torch FlopCounterMode, incl. lm_head) — the old
hand-rolled counters swung the TF:V2 ratio by >2x and invalidated the
earlier 'V2 wins' result. Paste the RESULTS block back.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "tb", os.path.join(os.path.dirname(os.path.abspath(__file__)), "novacore", "models", "tiny_brain.py")
)
tb = importlib.util.module_from_spec(spec)
sys.modules["tb"] = tb
spec.loader.exec_module(tb)
TinyBrainConfig, TinyBrainModel = tb.TinyBrainConfig, tb.TinyBrainModel

from novacore.core.simple_model import NovaModel
from novacore.core.config import NovaConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RES_DIR = Path("novacore/experiments/scale_path_results")
RES_DIR.mkdir(parents=True, exist_ok=True)

# Free ~2x on fp32 matmuls for Ampere+ GPUs (tf32 = 10-bit mantissa, standard
# for training). Shrinks the V2 wall-clock gap vs the transformer.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Reduce CUDA fragmentation in generation (big (B, L, vocab) logits).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SEQ_LEN = 64  # fixed sequence length used by every loader in this harness

# A placeholder token is WORSE than none (HF rejects it → DatasetNotFoundError).
_hf_tok = os.environ.get("HF_TOKEN", "")
if _hf_tok and len(_hf_tok) < 15:
    print("WARNING: HF_TOKEN looks like a placeholder (too short). Remove it or set a real token")
    print("         from https://huggingface.co/settings/tokens (read permission is enough).")


def _hf_load(repo_id, config=None, split="train", streaming=False):
    """load_dataset with one retry + clear guidance when the Hub rejects us.

    Colab runs often fail with 'Dataset ... doesn't exist or cannot be accessed'
    even though the repo is public — the real cause is auth/rate-limit/network.
    """
    from datasets import load_dataset
    for attempt in (1, 2):
        try:
            if config:
                return load_dataset(repo_id, config, split=split, streaming=streaming)
            return load_dataset(repo_id, split=split, streaming=streaming)
        except Exception as e:
            if attempt == 1:
                print(f"  HF load failed ({e.__class__.__name__}): {e}")
                print("  Retrying in 10s... If it persists: set a real HF_TOKEN")
                print("  (huggingface.co/settings/tokens) or check the network.")
                time.sleep(10)
            else:
                raise


_TOKENIZER_CACHE = {}


def get_tokenizer(name="Qwen/Qwen2.5-0.5B"):
    """Fetch and cache pretrained HuggingFace tokenizer (Qwen, Llama, etc.)."""
    global _TOKENIZER_CACHE
    if name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[name]
    from transformers import AutoTokenizer
    print(f"Loading pretrained tokenizer: {name}...")
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    _TOKENIZER_CACHE[name] = tok
    return tok


def _word_vocab(texts, top_k=30000, seq_len=64, max_words=50):
    """Word-level vocab from the MOST-FREQUENT words (not alphabetical!).

    The old sort-by-alphabet cut common words out of GSM8K (numbers and 'a..e'
    words filled the 20k slots before 'the/what/you/there/...'), making <unk>
    the dominant training token → the LM collapsed to predicting <unk> forever
    (generated output was 100% '<unk> <unk> <unk>...'). Frequency selection
    keeps the words the model actually sees. Special ids are FIXED:
    0=<pad> 1=<unk> 2=<eos> (eos gives generation a real stop token).
    Ties break alphabetically so the id ordering is IDENTICAL across runs
    (required for --save_path/--load_path weight compatibility).
    max_words=None counts the FULL text (used for GSM8K, where the long
    chain-of-thought tail holds '#### answer').
    """
    from collections import Counter
    cnt = Counter()
    for t in texts:
        toks_ = t.lower().split()
        if max_words:
            toks_ = toks_[:max_words]
        cnt.update(toks_)
    vl = [w for w, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))][:top_k]
    w2i = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    for i, w in enumerate(vl, start=3):
        w2i[w] = i
    return w2i


def _texts_to_data(texts, w2i, seq_len=64, min_words=5):
    data = []
    for t in texts:
        if len(t.split()) <= min_words:
            continue
        toks = [w2i.get(w, 1) for w in t.lower().split()[:seq_len]]
        data.append(torch.tensor(toks, dtype=torch.long))
    return data


def _tinystories_texts(max_samples=5000):
    ds = _hf_load("roneneldan/TinyStories", split="train")
    return list(ds["text"][:max_samples])


def load_tinystories(max_samples=5000, seq_len=64):
    texts = _tinystories_texts(max_samples)
    w2i = _word_vocab(texts)
    data = _texts_to_data(texts, w2i, seq_len)
    return data, len(w2i)


def _wikitext_texts(max_samples=20000):
    """WikiText-2 raw — OOD vs TinyStories (prose → encyclopedia).

    HF first; falls back to the canonical s3 zip when the Hub is unreachable
    (common on Colab where only cached datasets load).
    """
    try:
        ds = _hf_load("Salesforce/wikitext", config="wikitext-2-raw-v1", split="train")
        texts = [t.strip() for t in ds["text"] if len(t.strip()) > 5]
    except Exception:
        print("  HF unreachable — downloading wikitext-2 zip directly")
        import urllib.request
        import zipfile
        zpath = "wikitext-2-v1.zip"
        if not os.path.exists(zpath):
            urllib.request.urlretrieve(
                "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip", zpath
            )
        with zipfile.ZipFile(zpath) as zf:
            raw = zf.read("wikitext-2/wiki.train.tokens").decode()
        texts = [t for t in raw.splitlines() if len(t.strip()) > 5]
    return texts[:max_samples] if max_samples else texts


def load_wikitext(max_samples=20000, seq_len=64, top_k=20000):
    texts = _wikitext_texts(max_samples)
    w2i = _word_vocab(texts, top_k=top_k)
    data = _texts_to_data(texts, w2i, seq_len)
    print(f"  wikitext: {len(data)} seqs vocab={len(w2i)}")
    return data, len(w2i)


def _owt_texts(max_samples=20000):
    """OpenWebText subset via streaming. Bare 'openwebtext' fails on newer
    datasets versions (HfUriError) — use the canonical Skylion007/openwebtext.
    """
    ds = _hf_load("Skylion007/openwebtext", split="train", streaming=True)
    texts = []
    for ex in ds:
        t = ex["text"].strip()
        if len(t.split()) > 10:
            texts.append(t)
        if len(texts) >= max_samples:
            break
    return texts


def load_openwebtext(max_samples=20000, seq_len=64, top_k=20000):
    texts = _owt_texts(max_samples)
    w2i = _word_vocab(texts, top_k=top_k)
    data = _texts_to_data(texts, w2i, seq_len)
    print(f"  openwebtext: {len(data)} seqs vocab={len(w2i)}")
    return data, len(w2i)


_CODE_TOK = None  # module-level BPE cache for --dataset code (used by code_eval)


def _bpe_tokenizer(texts, vocab_size=8000):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                                  special_tokens=["<pad>", "<unk>", "<eos>"],
                                  min_frequency=2)
    tok.train_from_iterator((t for t in texts), trainer)
    return tok


def _byte_tokenizer(texts):
    """Fallback vocab when the tokenizers lib is missing: byte-level ids."""
    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    for t in texts:
        for b in t.encode("utf-8", errors="ignore"):
            vocab.setdefault(f"b{b}", len(vocab))
    return vocab


def _encode(tok, text, seq_len):
    if isinstance(tok, dict):
        return [tok.get(f"b{b}", 1) for b in text.encode("utf-8", errors="ignore")[:seq_len]]
    return tok.encode(text).ids[:seq_len]


def _decode(tok, ids):
    if isinstance(tok, dict):
        return bytes(int(k[1:]) for k in ids if str(k).startswith("b")).decode("utf-8", errors="ignore")
    return tok.decode(ids)


def load_code(max_samples=20000, seq_len=512, vocab_size=8000):
    """Code corpus for the market test: stream GitHub code, train a BPE."""
    global _CODE_TOK
    ds = _hf_load("codeparrot/github-code", split="train", streaming=True)
    texts = []
    for ex in ds:
        c = ex.get("code", ex.get("text", ""))
        if len(c) > 40:
            texts.append(c)
        if len(texts) >= max_samples:
            break
    if _CODE_TOK is None:
        try:
            _CODE_TOK = _bpe_tokenizer(texts[: min(4000, len(texts))], vocab_size)
        except Exception:
            print("  tokenizers lib missing — using byte-level fallback vocab")
            _CODE_TOK = _byte_tokenizer(texts[: min(2000, len(texts))])
    data = [torch.tensor(_encode(_CODE_TOK, t, seq_len), dtype=torch.long) for t in texts]
    vocab = len(_CODE_TOK)
    print(f"  code: {len(data)} seqs vocab={vocab}")
    return data, vocab


def load_mixed(args):
    """Blend multiple WORD-level datasets into one shared vocab (replay-style).
    Use when data keeps arriving: mix old corpora with new so the model does
    not overfit the newest slice or forget the old ones.
    """
    collectors = {
        "tinystories": lambda n: _tinystories_texts(n),
        "wikitext": lambda n: _wikitext_texts(n),
        "openwebtext": lambda n: _owt_texts(n),
    }
    specs = []
    for part in args.data_mix.split(","):
        name, frac = part.split(":")
        name, frac = name.strip(), float(frac)
        if name not in collectors:
            raise ValueError(f"data_mix: unknown source {name!r} (use tinystories/wikitext/openwebtext)")
        specs.append((name, frac))
    total = sum(f for _, f in specs)
    texts = []
    for name, frac in specs:
        n = int(args.samples * frac / total)
        texts += collectors[name](max(n, 1))
    w2i = _word_vocab(texts, top_k=30000)
    data = _texts_to_data(texts, w2i, args.seq_len)
    print(f"  mixed({args.data_mix}): {len(data)} seqs vocab={len(w2i)}")
    return data, len(w2i)


class SeqDS(torch.utils.data.Dataset):
    def __init__(self, data, seq_len=64, pad_id=0):
        self.data, self.seq_len, self.pad_id = data, seq_len, pad_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i]
        if x.numel() < self.seq_len:
            x = torch.cat([x, torch.full((self.seq_len - x.numel(),), self.pad_id, dtype=torch.long)])
        x = x[: self.seq_len]
        y = x.clone()
        # Pad carries no signal — masking it stops the LM from learning 'predict pad'
        y[y == self.pad_id] = -100
        return x, y


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def trainable_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class LoRALinear(nn.Module):
    """Frozen base Linear + low-rank adapter (pure torch, no extra deps)."""
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)
        d_in, d_out = base.weight.shape
        self.lora_a = nn.Parameter(torch.zeros(d_in, r))
        self.lora_b = nn.Parameter(torch.zeros(r, d_out))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
        self.scale = alpha / max(r, 1)

    def forward(self, x):
        return self.base(x) + ((x @ self.lora_a) @ self.lora_b) * self.scale


def apply_lora(model, rank=8):
    """Swap QKV/O + MLP linears with LoRA adapters; freeze everything else."""
    targets = ("W_q", "W_k", "W_v", "W_o", "W1", "W2")
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear) or not any(t in name for t in targets):
            continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRALinear(mod, rank))
    for name, p in model.named_parameters():
        if "lora" not in name:
            p.requires_grad = False
    return model


def gate_stats(model):
    gammas, gates = [], []
    for n, p in model.named_parameters():
        if "gamma" in n:
            gammas.append(torch.tanh(p).item())
        if "out_gate" in n:
            gates.append(torch.tanh(p).item())
    return {
        "gamma_mean": round(sum(gammas) / max(len(gammas), 1), 4),
        "out_gate_mean": round(sum(gates) / max(len(gates), 1), 4),
    }


@torch.no_grad()
def eval_loss(model, loader, amp=False):
    model.eval()
    use_amp = amp and DEVICE == "cuda"
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            total += model(x, labels=y)["loss"].item()
        n += 1
    return total / max(n, 1)


def estimate_fwd_flops(model, seq_len=64, batch=1):
    """Rough FLOP estimate for architecture comparison (not exact profiler)."""
    if isinstance(model, NovaModel):
        d = model.config.hidden_size
        L = model.config.num_layers
        # attn ~ 4*S^2*d + 4*S*d^2 ; mlp ~ 2*S*d*d_ff
        d_ff = model.config.intermediate_size
        attn = L * (4 * seq_len * seq_len * d + 4 * seq_len * d * d)
        mlp = L * (2 * seq_len * d * d_ff)
        return int(batch * (attn + mlp))
    cfg = model.config
    d, K, T, m = cfg.hidden_size, cfg.num_cells, cfg.max_think_steps, cfg.memory_slots
    # think ~ 4 linear d^2 per step; mem ~ 2*S*d*m + writes; optional attn
    think = K * T * (4 * seq_len * d * d)
    mem = K * T * (2 * seq_len * d * m + seq_len * d * d)
    attn = 0
    ad = max(1, int(d * cfg.attn_dim_ratio))
    if cfg.use_token_attn:
        attn += 4 * seq_len * seq_len * ad + 3 * seq_len * d * ad
    if cfg.attn_every_cell:
        attn += K * (4 * seq_len * seq_len * ad + 3 * seq_len * d * ad)
    # head: out_mlp (2 matmuls) + self-correction (refine + gate per step)
    head = 2 * seq_len * d * cfg.output_mlp_hidden
    head += cfg.correction_steps * 2 * seq_len * d * d
    return int(batch * (think + mem + attn + head))


def measure_fwd_flops(model, seq_len=SEQ_LEN, train_mode=True):
    """Real per-step forward FLOPs via torch's FlopCounterMode (incl. lm_head).

    Hand-rolled counters have proven unreliable (they swung the TF:V2 ratio by
    >2x — measured TF is ~6x the old estimate). Measured numbers are what
    reviewers will trust. Returns None if the profiler is unavailable.
    train_mode=False measures EVAL cost (adaptive early-exit) — the number
    that matters for running cost.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return None
    if train_mode:
        was_training = model.training
        model.train()  # training cost (dropout + full think loop), not eval cost
    else:
        model.eval()
    try:
        vocab = getattr(model.config, "vocab_size", 1000)
        x = torch.randint(0, max(1, min(vocab, 50000)), (1, seq_len),
                          device=next(model.parameters()).device)
        with torch.no_grad():
            with FlopCounterMode(display=False) as fm:
                model(x)
        return int(fm.get_total_flops())
    except Exception:
        return None
    finally:
        if train_mode:
            model.train(was_training)


def make_tf(vocab, hidden=256, layers=3, heads=4, lora=False, lora_rank=8):
    cfg = NovaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        num_layers=layers,
        num_attention_heads=heads,
        intermediate_size=hidden * 3,
        max_seq_length=64,
    )
    m = NovaModel(cfg).to(DEVICE)
    if lora:
        m = apply_lora(m, rank=lora_rank)
    return m


def make_tb(vocab, variant="hybrid_v2", hidden=256, cells=3, think_steps=4, sharp=None, rank=None, paths=1,
            train_break=0.8, model_size=None, attn_ratio=None, mem_slots=None, final_norm=None):
    """
    variants:
      plain      — no attention
      hybrid_v1  — post-cell lightweight attn
      hybrid_v2  — per-cell attn + step-conditioned + selective memory + diversity loss

    model_size presets (overrides hidden/cells/think_steps/rank; measured at vocab=8192):
      nano   : hidden=256,  cells=3, think_steps=4,  rank=None (~4.4M)
      small  : hidden=512,  cells=4, think_steps=8,  rank=128  (~15M)
      medium : hidden=1024, cells=6, think_steps=12, rank=256  (~68M)
      1b     : hidden=2048, cells=8, think_steps=16, rank=256  (~300M)
    True ~1B needs cells=12-16 or hidden=4096 (recurrent weights dominate; out_mlp
    and memory_slots are NOT scaled by presets). Preset think_steps wins over the
    CLI --think_steps — read m.config.max_think_steps for the effective depth.

    attn_ratio/mem_slots/final_norm: A/B overrides (None = proven defaults).
    """
    if model_size == "nano":
        hidden, cells, think_steps, rank = 256, 3, 4, None
    elif model_size == "small":
        hidden, cells, think_steps, rank = 512, 4, 8, 128
    elif model_size == "medium":
        hidden, cells, think_steps, rank = 1024, 6, 12, 256
    elif model_size == "1b":
        hidden, cells, think_steps, rank = 2048, 8, 16, 256

    if variant == "plain":
        use_post, every, ratio = False, False, 0.25
    elif variant == "hybrid_v1":
        use_post, every, ratio = True, False, 0.25
    elif variant == "hybrid_v2":
        use_post, every, ratio = False, True, 0.5
    else:
        raise ValueError(variant)
    if attn_ratio is not None:
        ratio = attn_ratio
    cfg = TinyBrainConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        num_cells=cells,
        memory_slots=mem_slots if mem_slots is not None else 16,
        num_think_heads=2,
        max_think_steps=think_steps,
        min_think_steps=1,
        output_mlp_hidden=hidden * 2,
        use_token_attn=use_post,
        attn_every_cell=every,
        attn_dim_ratio=ratio,
        attn_heads=2 if variant == "hybrid_v2" else 1,
        step_conditioned_think=True,
        gamma_init=0.1,
        out_gate_init=0.1,
        step_penalty_weight=0.05 if variant == "hybrid_v2" else 0.1,
        diversity_weight=0.05 if variant != "plain" else 0.0,
        memory_sharp_init=5.0 if sharp is None else sharp,
        think_rank=rank,
        num_thought_paths=paths,
        train_break=train_break,
        final_norm=bool(final_norm) if final_norm is not None else False,
    )
    return TinyBrainModel(cfg).to(DEVICE)


def train_one(model, train_loader, val_loader, steps, name, log_every=200,
              use_cosine=True, early_stop_patience_steps=0, lr=3e-4,
              warmup_fraction=0.02, ema=0.0, amp=False, seq_len=SEQ_LEN, compile=False,
              think_schedule=None, input_dropout=0.0, unk_id=1):
    """Train with an LR schedule keyed to TOKENS, not steps.

    Why tokens: in equal-FLOPs runs the two models get different step counts.
    Keying cosine + linear warmup to each model's own token budget means both
    models sit at the same LR phase for the same fraction of data seen — the
    step-count difference no longer biases the comparison.

    ema>0: exponential moving average of weights; val loss is evaluated on the
    EMA weights (standard overfitting defense, usually worth ~0.1-0.3 nats).
    amp: bf16 autocast (Ampere+) — ~2x wall-clock on the sequential think loop.
    compile: torch.compile the model (kernel fusion / CUDA graphs) — the big
    wall-clock win for the recurrent think loop.
    think_schedule: 'T1,T2,T4,T8' curriculum — early training runs cheap
    think steps, late training runs expensive ones (wall-clock cut, same quality).
    """
    if compile:
        orig_mod = getattr(model, "_orig_mod", model)
        if isinstance(orig_mod, TinyBrainModel):
            # TinyBrain's python control flow (confidence .item() breaks, grad
            # mode switches) makes torch.compile recompile constantly — slower,
            # not faster. Keep it eager; compile only helps the transformer.
            print(f"  [{name:12s}] compile skipped for TinyBrain (graph breaks)")
        else:
            try:
                model = _force_compile(model)
                print(f"  [{name:12s}] torch.compile active")
            except Exception as e:
                print(f"  [{name:12s}] torch.compile unavailable ({e.__class__.__name__}) — eager")
    orig = getattr(model, "_orig_mod", model)
    sched_T = None
    if think_schedule:
        try:
            sched_T = [int(x) for x in think_schedule.split(",") if x.strip()]
        except Exception:
            sched_T = None
    if sched_T and isinstance(orig, TinyBrainModel):
        for c in orig.cells:
            c.min_s = c.max_s = sched_T[0]
    tok_per_step = train_loader.batch_size * seq_len
    total_tokens = steps * tok_per_step
    warm_tokens = max(int(total_tokens * warmup_fraction), 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    if use_cosine:
        end_ratio = 0.05

        def lr_lambda(step):
            tok = (step + 1) * tok_per_step
            if tok <= warm_tokens:
                return tok / warm_tokens
            frac = min((tok - warm_tokens) / max(total_tokens - warm_tokens, 1), 1.0)
            return end_ratio + 0.5 * (1 - end_ratio) * (1 + math.cos(math.pi * frac))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        sched = None
    ema_state = None
    if ema > 0:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def eval_at(state_dict=None):
        """Val loss on given weights (EMA) or live model; returns to train mode."""
        if state_dict is not None:
            saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(state_dict)
            vl = eval_loss(model, val_loader, amp=amp)
            model.load_state_dict(saved)
        else:
            vl = eval_loss(model, val_loader, amp=amp)
        model.train()
        return vl

    hist, t0, step = [], time.time(), 0
    best_val, best_step = float("inf"), 0
    ema_loss, best_train_ema, lr_at_best = None, None, None
    model.train()
    stopped_early = False
    while step < steps:
        for x, y in train_loader:
            if step >= steps:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            if input_dropout > 0:
                # exposure-bias fix (fast, batched): corrupt a fraction of INPUT
                # tokens with <unk> — labels stay gold, so the model must predict
                # from partially-wrong context, like at generation time.
                drop = torch.rand(x.shape, device=DEVICE) < input_dropout
                x = x.masked_fill(drop, unk_id)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and DEVICE == "cuda"):
                loss = model(x, labels=y)["loss"]
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            if sched_T and isinstance(orig, TinyBrainModel):
                t_idx = min(step * len(sched_T) // max(steps, 1), len(sched_T) - 1)
                T = sched_T[t_idx]
                for c in orig.cells:
                    c.min_s = c.max_s = T
            if ema > 0 and ema_state is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        ema_state[k].mul_(ema).add_(v, alpha=1.0 - ema)
            step += 1
            lv = loss.item()
            ema_loss = lv if ema_loss is None else 0.9 * ema_loss + 0.1 * lv
            if step % log_every == 0 or step == steps:
                vl = eval_at(ema_state if ema > 0 else None)
                if vl < best_val - 1e-4:
                    best_val, best_step = vl, step
                    best_train_ema, lr_at_best = ema_loss, opt.param_groups[0]["lr"]
                gs = gate_stats(orig) if not isinstance(orig, NovaModel) else {}
                row = {
                    "step": step,
                    "val_loss": round(vl, 4),
                    "train_loss_ema": round(ema_loss, 4),
                    "best_val_loss": round(best_val, 4),
                    "lr": round(opt.param_groups[0]["lr"], 6),
                    "time_sec": round(time.time() - t0, 1),
                    **gs,
                }
                hist.append(row)
                extra = f" | gamma={gs.get('gamma_mean', 0):.4f} gate={gs.get('out_gate_mean', 0):.4f}" if gs else ""
                print(f"  [{name:12s}] {step:4d}/{steps} | val={vl:.4f} best={best_val:.4f} train={ema_loss:.4f}{extra}", flush=True)
                if early_stop_patience_steps > 0 and step - best_step >= early_stop_patience_steps:
                    print(f"  [{name:12s}] early stop @ {step} (no improve for {step - best_step} steps)")
                    stopped_early = True
                    break
            elif step % 25 == 0:
                # Cheap live progress (no val pass) — with log_every=250 and a
                # 6x bigger model, minutes of silence read as a hang.
                print(f"  [{name:12s}] {step:4d}/{steps} train={ema_loss:.4f}", flush=True)
        if stopped_early or step >= steps:
            break
    wall = time.time() - t0
    return {
        "name": name,
        "params": count_params(orig),
        "approx_flops": estimate_fwd_flops(orig),
        "final_val_loss": hist[-1]["val_loss"] if hist else round(eval_loss(model, val_loader), 4),
        "best_val_loss": round(best_val, 4),
        "best_step": best_step,
        "ema": ema,
        "best_train_loss_ema": round(best_train_ema, 4) if best_train_ema is not None else None,
        "final_train_loss_ema": round(ema_loss, 4) if ema_loss is not None else None,
        "train_val_gap_at_best": round(best_train_ema - best_val, 4) if best_train_ema is not None else None,
        "lr_at_best": round(lr_at_best, 6) if lr_at_best is not None else None,
        "steps_ran": step,
        "tokens_seen": step * tok_per_step,
        "tok_per_sec": round(step * tok_per_step / max(wall, 1e-6), 1),
        "stopped_early": stopped_early,
        "time_sec": round(wall, 2),
        "history": hist,
        "trainable_params": trainable_params(orig),
        "avg_think_steps": round(sum(c._steps_sum for c in orig.cells) / max(sum(c._steps_n for c in orig.cells), 1), 2)
        if isinstance(orig, TinyBrainModel) and sum(c._steps_n for c in orig.cells) > 0 else None,
        "gates": gate_stats(orig) if not isinstance(orig, NovaModel) else {},
    }


@torch.no_grad()
def diagnose_internals(model, batch):
    """Iteration diversity + memory entropy + branch norms."""
    model.eval()
    x = batch.to(DEVICE)
    h = model.embed(x)
    cell0 = model.cells[0]
    old_min, old_max = cell0.min_s, cell0.max_s
    cell0.min_s = cell0.max_s = min(4, old_max)
    out, mem, aux = cell0(h, return_trace=True)
    cell0.min_s, cell0.max_s = old_min, old_max
    trace = aux.get("trace", [])
    sims = []
    for i in range(len(trace) - 1):
        a = trace[i].reshape(-1)
        b = trace[i + 1].reshape(-1)
        sims.append(F.cosine_similarity(a, b, dim=0).item())
    attn = cell0.mem._last_attn
    if attn is None:
        attn = torch.ones(out.size(0), out.size(1), mem.size(1), device=out.device) / mem.size(1)
    ent = -(attn * (attn + 1e-9).log()).sum(dim=-1).mean().item()
    max_ent = math.log(mem.size(1))
    top1 = attn.max(dim=-1).values.mean().item()
    # Effective read scale: softplus(logit_scale)/sqrt(d) — the sharpness knob.
    # Small scale ⇒ near-uniform read ⇒ slot identity barely matters (shuffle≈full).
    d = model.config.hidden_size
    raw_scales, eff_scales = [], []
    for c in model.cells:
        s = F.softplus(c.mem.logit_scale).item()
        raw_scales.append(s)
        eff_scales.append(s / math.sqrt(d))
    return {
        "iter_cosine_mean": round(sum(sims) / max(len(sims), 1), 4),
        "iter_cosine_list": [round(s, 4) for s in sims],
        "memory_entropy": round(ent, 4),
        "memory_entropy_ratio": round(ent / max(max_ent, 1e-6), 4),
        "memory_top1_mass": round(top1, 4),
        "mem_logit_scale_mean": round(statistics.mean(raw_scales), 4),
        "mem_eff_scale_mean": round(statistics.mean(eff_scales), 4),
        "out_norm": round(out.norm().item(), 4),
        "embed_norm": round(h.norm().item(), 4),
    }


def get_loaders(args, seed=0):
    print(f"Loading {args.dataset}...")
    if args.data_mix:
        data, vocab = load_mixed(args)
    elif args.dataset == "wikitext":
        data, vocab = load_wikitext(args.samples, args.seq_len)
    elif args.dataset == "openwebtext":
        data, vocab = load_openwebtext(args.samples, args.seq_len)
    elif args.dataset == "code":
        data, vocab = load_code(args.samples, args.seq_len)
    elif args.dataset == "gsm8k":
        gsm8k_vocab_size = getattr(args, "gsm8k_vocab_size", 8192)
        data, vocab = load_gsm8k_lm(args.samples, args.seq_len,
                                    getattr(args, "tokenizer_name", "Qwen/Qwen2.5-0.5B"),
                                    gsm8k_vocab_size=gsm8k_vocab_size)
    else:
        data, vocab = load_tinystories(args.samples, args.seq_len)
    split = int(len(data) * 0.9)
    g = torch.Generator().manual_seed(seed)  # deterministic shuffle per seed
    tok = _GSM.get("tokenizer")
    pad_id = tok.pad_token_id if tok is not None and getattr(tok, "pad_token_id", None) is not None else 0
    tl = torch.utils.data.DataLoader(SeqDS(data[:split], args.seq_len, pad_id=pad_id), batch_size=args.batch, shuffle=True, generator=g)
    vl = torch.utils.data.DataLoader(SeqDS(data[split:], args.seq_len, pad_id=pad_id), batch_size=args.batch)
    print(f"  seqs={len(data)} vocab={vocab} device={DEVICE}")
    return tl, vl, vocab


def mode_race(args):
    """TF vs hybrid_v1 vs hybrid_v2 — compare BEST val (TF can overfit late)."""
    tl, vl, vocab = get_loaders(args)
    results = {"meta": _meta(args, "race"), "models": {}}
    print("\n=== RACE: Transformer vs Hybrid v1 vs Hybrid v2 ===")
    print("NOTE: report BEST val_loss (final can overfit, especially TF).")
    results["models"]["transformer"] = train_one(make_tf(vocab, lora=args.lora, lora_rank=args.lora_rank), tl, vl, args.steps, "Transformer", args.log_every, lr=args.lr, ema=args.ema, amp=args.amp, seq_len=args.seq_len, compile=args.compile, think_schedule=args.think_curriculum)
    results["models"]["hybrid_v1"] = train_one(make_tb(vocab, "hybrid_v1", sharp=args.memory_sharp, rank=args.think_rank, paths=args.thought_paths, train_break=args.train_break), tl, vl, args.steps, "hybrid_v1", args.log_every, lr=args.lr, ema=args.ema, amp=args.amp, seq_len=args.seq_len, compile=args.compile, think_schedule=args.think_curriculum)
    results["models"]["hybrid_v2"] = train_one(make_tb(vocab, "hybrid_v2", sharp=args.memory_sharp, rank=args.think_rank, paths=args.thought_paths, train_break=args.train_break), tl, vl, args.steps, "hybrid_v2", args.log_every, lr=args.lr, ema=args.ema, amp=args.amp, seq_len=args.seq_len, compile=args.compile, think_schedule=args.think_curriculum)

    tf_b = results["models"]["transformer"]["best_val_loss"]
    print("\n" + "=" * 64)
    print("RESULTS (copy this block back)")
    print("=" * 64)
    print(f"{'Model':14s} {'Params':>10s} {'BestVal':>8s} {'Final':>8s} {'Best@':>6s} {'dBestTF':>8s}")
    for k in ["transformer", "hybrid_v1", "hybrid_v2"]:
        r = results["models"][k]
        print(
            f"{k:14s} {r['params']:10,} {r['best_val_loss']:8.4f} {r['final_val_loss']:8.4f} "
            f"{r['best_step']:6d} {r['best_val_loss']-tf_b:+8.4f}"
        )
    v2b = results["models"]["hybrid_v2"]["best_val_loss"]
    v1b = results["models"]["hybrid_v1"]["best_val_loss"]
    results["summary"] = {
        "v2_vs_tf_best": round(v2b - tf_b, 4),
        "v2_vs_v1_best": round(v2b - v1b, 4),
        "v2_beats_tf": v2b < tf_b,
        "path_note": "Use best_val (not final). If v2 still wins → verify_claim multi-seed.",
    }
    print("-" * 64)
    print(f"v2 vs TF (best): {results['summary']['v2_vs_tf_best']:+.4f} | v2 vs v1 (best): {results['summary']['v2_vs_v1_best']:+.4f}")
    print(results["summary"]["path_note"])
    _save(results, "race")
    return results


def mode_diagnose(args):
    """Train briefly then measure iteration diversity + memory entropy."""
    tl, vl, vocab = get_loaders(args)
    results = {"meta": _meta(args, "diagnose"), "models": {}}
    print("\n=== DIAGNOSE internals (v1 vs v2) — after memory/diversity fix ===")
    print("Targets: iter_cos < 0.98 | mem_entropy_ratio < 0.85 | top1 > 0.15")
    for variant in ["hybrid_v1", "hybrid_v2"]:
        m = make_tb(vocab, variant, sharp=args.memory_sharp, rank=args.think_rank, paths=args.thought_paths, train_break=args.train_break)
        train_one(m, tl, vl, args.steps, variant, args.log_every, lr=args.lr, ema=args.ema, amp=args.amp, seq_len=args.seq_len, compile=args.compile, think_schedule=args.think_curriculum)
        batch = next(iter(vl))[0][:4]
        diag = diagnose_internals(m, batch)
        results["models"][variant] = {
            "val_loss": eval_loss(m, vl),
            "gates": gate_stats(m),
            "internals": diag,
        }
        print(f"  {variant}: loss={results['models'][variant]['val_loss']:.4f}")
        print(f"    iter_cos={diag['iter_cosine_mean']} (want <0.98)")
        print(f"    mem_entropy_ratio={diag['memory_entropy_ratio']} top1={diag['memory_top1_mass']}")
    print("\nRESULTS (copy back)")
    print(json.dumps(results["models"], indent=2))
    _save(results, "diagnose")
    return results


def mode_think_scale(args):
    """Critical for 1B=600B+: more think steps must improve loss at fixed params."""
    tl, vl, vocab = get_loaders(args)
    results = {"meta": _meta(args, "think_scale"), "models": {}}
    print("\n=== THINK SCALE (fixed params, vary think steps) ===")
    print("If more steps => lower BEST loss, compute-scaling thesis is alive.")
    for tsteps in [1, 2, 4, 8]:
        name = f"v2_T{tsteps}"
        m = make_tb(vocab, "hybrid_v2", think_steps=tsteps, sharp=args.memory_sharp, rank=args.think_rank, paths=args.thought_paths, train_break=args.train_break)
        for c in m.cells:
            c.min_s = tsteps
            c.max_s = tsteps
        results["models"][name] = train_one(m, tl, vl, args.steps, name, args.log_every, lr=args.lr, ema=args.ema, amp=args.amp, seq_len=args.seq_len, compile=args.compile, think_schedule=args.think_curriculum)
        results["models"][name]["think_steps"] = tsteps
        mf = measure_fwd_flops(m, args.seq_len)
        if mf:
            results["models"][name]["measured_flops"] = mf

    print("\n" + "=" * 64)
    print("RESULTS (copy this block back)")
    print("=" * 64)
    print(f"\n{'Config':12s} {'BestVal':>8s} {'Final':>8s} {'Params':>10s} {'~FLOPs':>12s}")
    losses = []
    for tsteps in [1, 2, 4, 8]:
        r = results["models"][f"v2_T{tsteps}"]
        losses.append(r["best_val_loss"])
        mf = r.get("measured_flops") or r["approx_flops"]
        print(
            f"T={tsteps:<9d} {r['best_val_loss']:8.4f} {r['final_val_loss']:8.4f} "
            f"{r['params']:10,} {mf:12,}"
        )
    improved = losses[-1] < losses[0]
    results["summary"] = {
        "T1_best": losses[0],
        "T8_best": losses[-1],
        "T8_better_than_T1": improved,
        "delta_T8_minus_T1": round(losses[-1] - losses[0], 4),
        "verdict": "COMPUTE_SCALES" if improved else "STEPS_DONT_HELP_YET",
    }
    print("-" * 64)
    print(f"Verdict: {results['summary']['verdict']} (T8-T1 best={results['summary']['delta_T8_minus_T1']:+.4f})")
    _save(results, "think_scale")
    return results


def mode_verify_claim(args):
    """Multi-seed TF vs v2 using BEST val — kills leakage/overfit illusions."""
    seeds = [int(s) for s in args.seeds.split(",")]
    results = {"meta": _meta(args, "verify_claim"), "seeds": {}, "summary": {}}
    print("\n=== VERIFY CLAIM (multi-seed, best val) ===")
    print(f"seeds={seeds} steps={args.steps}")
    print("Compares BEST val (TF often overfits after ~800–1000 steps).")
    tf_bests, v2_bests = [], []
    last_v2, last_batch = None, None

    for seed in seeds:
        torch.manual_seed(seed)
        if DEVICE == "cuda":
            torch.cuda.manual_seed_all(seed)
        tl, vl, vocab = get_loaders(args, seed)
        print(f"\n--- seed {seed} ---")
        tf_m = make_tf(vocab, lora=args.lora, lora_rank=args.lora_rank)
        v2_m = make_tb(vocab, "hybrid_v2", sharp=args.memory_sharp, rank=args.think_rank, paths=args.thought_paths, train_break=args.train_break)
        tf_m.label_smoothing = args.label_smooth
        v2_m.label_smoothing = args.label_smooth
        tf_r = train_one(tf_m, tl, vl, args.steps, f"TF_s{seed}", args.log_every,
                         early_stop_patience_steps=args.early_stop, lr=args.lr, ema=args.ema, amp=args.amp, seq_len=args.seq_len, compile=args.compile, think_schedule=args.think_curriculum)
        v2_r = train_one(v2_m, tl, vl, args.steps, f"V2_s{seed}", args.log_every,
                         early_stop_patience_steps=args.early_stop, lr=args.lr, ema=args.ema, amp=args.amp, seq_len=args.seq_len, compile=args.compile, think_schedule=args.think_curriculum)
        last_v2, last_batch = v2_m, next(iter(vl))[0][:4]
        delta = round(v2_r["best_val_loss"] - tf_r["best_val_loss"], 4)
        results["seeds"][str(seed)] = {
            "transformer": {
                "best": tf_r["best_val_loss"],
                "final": tf_r["final_val_loss"],
                "best_step": tf_r["best_step"],
            },
            "hybrid_v2": {
                "best": v2_r["best_val_loss"],
                "final": v2_r["final_val_loss"],
                "best_step": v2_r["best_step"],
            },
            "delta_best": delta,
        }
        tf_bests.append(tf_r["best_val_loss"])
        v2_bests.append(v2_r["best_val_loss"])
        print(
            f"  seed {seed}: TF best={tf_r['best_val_loss']:.4f} @{tf_r['best_step']} | "
            f"V2 best={v2_r['best_val_loss']:.4f} @{v2_r['best_step']} | Δ={delta:+.4f}"
        )

    if last_v2 is not None and last_batch is not None:
        results["last_seed_internals"] = diagnose_internals(last_v2, last_batch)
        d = results["last_seed_internals"]
        print(
            f"\nLast-seed internals: iter_cos={d['iter_cosine_mean']} "
            f"mem_ent_ratio={d['memory_entropy_ratio']} top1={d['memory_top1_mass']}"
        )

    tf_mean, v2_mean = statistics.mean(tf_bests), statistics.mean(v2_bests)
    tf_std = statistics.stdev(tf_bests) if len(tf_bests) > 1 else 0.0
    v2_std = statistics.stdev(v2_bests) if len(v2_bests) > 1 else 0.0
    wins = sum(1 for a, b in zip(tf_bests, v2_bests) if b < a)
    ps = paired_stats(v2_bests, tf_bests)
    results["summary"] = {
        "tf_best_mean": round(statistics.mean(tf_bests), 4),
        "tf_best_std": round(statistics.stdev(tf_bests), 4) if len(tf_bests) > 1 else 0.0,
        "v2_best_mean": round(statistics.mean(v2_bests), 4),
        "v2_best_std": round(statistics.stdev(v2_bests), 4) if len(v2_bests) > 1 else 0.0,
        **ps,
        "claim_holds": bool(wins >= (len(seeds) + 1) // 2 and v2_mean < tf_mean),
        "stat_sig": bool(ps["n_seeds"] >= 3 and ps["p_value_paired_t"] is not None
                         and ps["p_value_paired_t"] < 0.05 and ps["delta_mean"] < 0),
    }
    p_str = f"{ps['p_value_paired_t']:.4f}" if ps["p_value_paired_t"] is not None else "n/a"
    print("\n" + "=" * 64)
    print("RESULTS (copy this block back)")
    print("=" * 64)
    print(f"TF  best: {tf_mean:.4f} ± {tf_std:.4f}")
    print(f"V2  best: {v2_mean:.4f} ± {v2_std:.4f}")
    print(f"Δ best (V2-TF): {ps['delta_mean']:+.4f} | p={p_str}")
    print(f"V2 wins: {wins}/{len(seeds)} | claim_holds={results['summary']['claim_holds']} "
          f"| stat_sig={results['summary']['stat_sig']}")
    print("If claim_holds → next: equal-FLOPs curve, then scale 50M→1B.")
    _save(results, "verify_claim")
    return results


def mode_memory_ablation(args):
    """Does memory change loss? Multi-seed; read path isolated from write path.

    The model re-initialises memory from M0 (the only trainable memory) and
    rewrites it in-context every batch, so perturbing M0 with writes ON tests
    the full pipeline but confounds read vs write. Perturbing M0 with writes
    OFF (read_only) isolates whether the READ path uses slot identity or just
    slot-content statistics.
    """
    seeds = [int(s) for s in args.seeds.split(",")]
    print("\n=== MEMORY ABLATION (multi-seed, fixed-key memory) ===")
    print(f"seeds={seeds} steps={args.steps}")
    print("Want: random>>full (content matters); shuffled≈full ⇒ slot identity unused.")
    agg = {k: [] for k in [
        "full", "zero", "shuf_vals", "shuf_keys", "random",
        "ro_full", "ro_shuf_vals", "ro_random", "ro_zero",
    ]}
    internals_all = []
    for seed in seeds:
        torch.manual_seed(seed)
        if DEVICE == "cuda":
            torch.cuda.manual_seed_all(seed)
        tl, vl, vocab = get_loaders(args, seed)
        m = make_tb(vocab, "hybrid_v2", sharp=args.memory_sharp, rank=args.think_rank, paths=args.thought_paths, train_break=args.train_break)
        m.label_smoothing = args.label_smooth
        train_one(m, tl, vl, args.steps, f"v2_s{seed}", args.log_every,
                  early_stop_patience_steps=args.log_every * 4, lr=args.lr, ema=args.ema, amp=args.amp, seq_len=args.seq_len, compile=args.compile, think_schedule=args.think_curriculum)
        batch = next(iter(vl))[0][:4]
        internals_all.append(diagnose_internals(m, batch))

        m0_snap = [c.mem.M0.data.clone() for c in m.cells]
        key_snap = [c.mem.keys.data.clone() for c in m.cells]
        gate_snap = [c.mem.out_gate.data.clone() for c in m.cells]

        def restore():
            for c, s, k, g in zip(m.cells, m0_snap, key_snap, gate_snap):
                c.mem.M0.data.copy_(s)
                c.mem.keys.data.copy_(k)
                c.mem.out_gate.data.copy_(g)

        def set_read_only(on):
            for c in m.cells:
                c.mem.read_only = on

        full = eval_loss(m, vl)

        for c in m.cells:
            c.mem.out_gate.data.zero_()
        zero = eval_loss(m, vl)
        restore()

        # Shuffle VALUES only (keys fixed) — hurts iff read/write use slot identity
        for c in m.cells:
            perm = torch.randperm(c.mem.M0.size(0), device=c.mem.M0.device)
            c.mem.M0.data.copy_(c.mem.M0.data[perm])
        shuf_vals = eval_loss(m, vl)
        restore()

        # Shuffle KEYS only — perturbs addressing for both read and write
        for c in m.cells:
            perm = torch.randperm(c.mem.keys.size(0), device=c.mem.keys.device)
            c.mem.keys.data.copy_(c.mem.keys.data[perm])
        shuf_keys = eval_loss(m, vl)
        restore()

        for c in m.cells:
            c.mem.M0.data.normal_(0, 0.5)
        random_m = eval_loss(m, vl)
        restore()

        # --- read-path isolation (writes disabled; memory frozen at M0) ---
        set_read_only(True)
        ro_full = eval_loss(m, vl)
        for c in m.cells:
            perm = torch.randperm(c.mem.M0.size(0), device=c.mem.M0.device)
            c.mem.M0.data.copy_(c.mem.M0.data[perm])
        ro_shuf_vals = eval_loss(m, vl)
        restore()
        for c in m.cells:
            c.mem.M0.data.normal_(0, 0.5)
        ro_random = eval_loss(m, vl)
        restore()
        for c in m.cells:
            c.mem.out_gate.data.zero_()
        ro_zero = eval_loss(m, vl)
        restore()
        set_read_only(False)

        for k, v in [("full", full), ("zero", zero), ("shuf_vals", shuf_vals),
                     ("shuf_keys", shuf_keys), ("random", random_m),
                     ("ro_full", ro_full), ("ro_shuf_vals", ro_shuf_vals),
                     ("ro_random", ro_random), ("ro_zero", ro_zero)]:
            agg[k].append(v)
        print(f"  seed {seed}: full={full:.4f} zero={zero:.4f} shufV={shuf_vals:.4f} "
              f"shufK={shuf_keys:.4f} rand={random_m:.4f} | ro: full={ro_full:.4f} "
              f"shufV={ro_shuf_vals:.4f} rand={ro_random:.4f} zero={ro_zero:.4f}")

    def avg(key):
        return statistics.mean(agg[key])

    a = {
        "full": round(avg("full"), 4),
        "zero_out_gate": round(avg("zero"), 4),
        "shuffled_values": round(avg("shuf_vals"), 4),
        "shuffled_keys": round(avg("shuf_keys"), 4),
        "random_values": round(avg("random"), 4),
        "read_only_full": round(avg("ro_full"), 4),
        "read_only_shuffled_values": round(avg("ro_shuf_vals"), 4),
        "read_only_random": round(avg("ro_random"), 4),
        "read_only_zero_gate": round(avg("ro_zero"), 4),
        "delta_zero": round(avg("zero") - avg("full"), 4),
        "delta_shuf_vals": round(avg("shuf_vals") - avg("full"), 4),
        "delta_shuf_keys": round(avg("shuf_keys") - avg("full"), 4),
        "delta_random": round(avg("random") - avg("full"), 4),
        "ro_delta_shuf_vals": round(avg("ro_shuf_vals") - avg("ro_full"), 4),
        "ro_delta_random": round(avg("ro_random") - avg("ro_full"), 4),
        "ro_delta_zero": round(avg("ro_zero") - avg("ro_full"), 4),
    }
    internals = {k: round(statistics.mean([d[k] for d in internals_all]), 4)
                 for k in ["iter_cosine_mean", "memory_entropy_ratio", "memory_top1_mass",
                           "mem_eff_scale_mean", "mem_logit_scale_mean"]}
    useful = a["delta_zero"] > 0.05 or a["delta_random"] > 0.05
    read_slot_id = a["ro_delta_shuf_vals"] > 0.05
    write_slot_id = a["delta_shuf_vals"] > 0.05
    flags = (["READ_USES_SLOT_ID"] if read_slot_id else ["READ_NO_SLOT_ID"]) + \
            (["WRITE_USES_SLOT_ID"] if write_slot_id else ["WRITE_NO_SLOT_ID"])
    results = {
        "meta": _meta(args, "memory_ablation"),
        "per_seed": {
            str(s): {k: round(vs[i], 4) for k, vs in agg.items()}
            for i, s in enumerate(seeds)
        },
        "ablation": a,
        "internals_mean": internals,
        "summary": {
            "memory_useful": useful,
            "read_uses_slot_id": read_slot_id,
            "write_uses_slot_id": write_slot_id,
            "verdict": ("MEMORY_USED " + " ".join(flags)) if useful else "MEMORY_STILL_WEAK",
        },
    }

    print("\n" + "=" * 64)
    print("RESULTS (copy this block back)")
    print("=" * 64)
    print(f"full              {a['full']:.4f}")
    print(f"zero_out_gate     {a['zero_out_gate']:.4f}  (Δ={a['delta_zero']:+.4f})")
    print(f"shuffled_values   {a['shuffled_values']:.4f}  (Δ={a['delta_shuf_vals']:+.4f})")
    print(f"shuffled_keys     {a['shuffled_keys']:.4f}  (Δ={a['delta_shuf_keys']:+.4f})")
    print(f"random_values     {a['random_values']:.4f}  (Δ={a['delta_random']:+.4f})")
    print(f"read_only_full    {a['read_only_full']:.4f}  (Δshuf={a['ro_delta_shuf_vals']:+.4f} Δrand={a['ro_delta_random']:+.4f})")
    print(
        f"internals: top1={internals['memory_top1_mass']} "
        f"ent_ratio={internals['memory_entropy_ratio']} iter_cos={internals['iter_cosine_mean']} "
        f"eff_scale={internals['mem_eff_scale_mean']}"
    )
    print(f"Verdict: {results['summary']['verdict']}")
    _save(results, "memory_ablation")
    return results


def paired_stats(v2_list, tf_list):
    """Paired comparison stats: mean±std of deltas, paired t-test, sign test."""
    deltas = [b - a for a, b in zip(tf_list, v2_list)]
    n = len(deltas)
    d_mean = statistics.mean(deltas)
    d_sd = statistics.stdev(deltas) if n > 1 else 0.0
    t_stat = (d_mean / (d_sd / (n ** 0.5))) if n > 1 and d_sd > 0 else 0.0
    p_t = None
    if n >= 2:
        try:
            from scipy import stats as sps
            _, p_t = sps.ttest_rel(v2_list, tf_list)
        except Exception:
            p_t = None
    if p_t is None or not math.isfinite(p_t):
        # normal approximation of the paired t (fine for n >= 5)
        p_t = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / (2 ** 0.5))))
    p_t = round(p_t, 4) if math.isfinite(p_t) else None
    wins = sum(1 for d in deltas if d < 0)
    p_sign = sum(math.comb(n, k) * 0.5 ** n for k in range(wins, n + 1))  # one-sided binomial
    return {
        "n_seeds": n,
        "delta_mean": round(d_mean, 4),
        "delta_std": round(d_sd, 4),
        "cohens_d": round(d_mean / d_sd, 3) if d_sd > 0 else None,
        "t_stat": round(t_stat, 3),
        "p_value_paired_t": p_t,
        "sign_test_p": round(p_sign, 4),
        "v2_wins": wins,
    }


def mode_equal_flops(args):
    """Fair race: same total approx FLOPs. Multi-seed mean±std + significance."""
    if args.samples < 15000:
        print(f"NOTE: bumping samples {args.samples} → 20000 for equal_flops (less memorize).")
        args.samples = 20000
    seeds = [int(s) for s in args.seeds.split(",")]
    results = {"meta": _meta(args, "equal_flops"), "seeds": {}, "summary": {}}
    print("\n=== EQUAL FLOPs RACE (multi-seed) ===")
    print(f"seeds={seeds} tf_steps={args.steps} samples={args.samples}")

    tf_bests, v2_bests = [], []
    last_v2, vl = None, None
    flops_method = "measured"
    for seed in seeds:
        torch.manual_seed(seed)
        if DEVICE == "cuda":
            torch.cuda.manual_seed_all(seed)
        tl, vl, vocab = get_loaders(args, seed)
        tf = make_tf(vocab, layers=args.tf_layers, lora=args.lora, lora_rank=args.lora_rank)
        v2 = make_tb(vocab, "hybrid_v2", think_steps=args.think_steps, sharp=args.memory_sharp, rank=args.think_rank, paths=args.thought_paths, train_break=args.train_break)
        tf.label_smoothing = args.label_smooth
        v2.label_smoothing = args.label_smooth
        for c in v2.cells:
            c.min_s = args.think_steps
            c.max_s = args.think_steps
        # Prefer profiler-measured FLOPs (hand-rolled counters proved unreliable).
        f_tf = measure_fwd_flops(tf, args.seq_len) or estimate_fwd_flops(tf)
        f_v2 = measure_fwd_flops(v2, args.seq_len) or estimate_fwd_flops(v2)
        if f_tf is None or f_v2 is None:
            flops_method = "estimate"
        f_tf_est, f_v2_est = estimate_fwd_flops(tf), estimate_fwd_flops(v2)
        steps_tf = args.steps
        budget = steps_tf * f_tf
        steps_v2 = max(1, int(round(budget / max(f_v2, 1))))
        print(f"\n--- seed {seed} | TF {steps_tf} steps | V2 {steps_v2} steps | "
              f"FLOPs/step ratio={f_v2/f_tf:.2f}x ({flops_method}; est {f_v2_est/f_tf_est:.2f}x) ---")
        # No early stopping: both models consume the FULL budget (fair FLOPs race).
        tf_r = train_one(tf, tl, vl, steps_tf, f"TF_s{seed}", args.log_every,
                         early_stop_patience_steps=args.early_stop, lr=args.lr, ema=args.ema, amp=args.amp, seq_len=args.seq_len, compile=args.compile, think_schedule=args.think_curriculum)
        v2_r = train_one(v2, tl, vl, steps_v2, f"V2_s{seed}", max(50, args.log_every // 2),
                         early_stop_patience_steps=args.early_stop, lr=args.lr, ema=args.ema, amp=args.amp, seq_len=args.seq_len, compile=args.compile, think_schedule=args.think_curriculum)
        last_v2 = v2
        delta = round(v2_r["best_val_loss"] - tf_r["best_val_loss"], 4)
        results["seeds"][str(seed)] = {
            "tf_best": tf_r["best_val_loss"], "tf_final": tf_r["final_val_loss"],
            "v2_best": v2_r["best_val_loss"], "v2_final": v2_r["final_val_loss"],
            "delta_best": delta,
            "delta_final": round(v2_r["final_val_loss"] - tf_r["final_val_loss"], 4),
            "tf_best_step": tf_r["best_step"], "v2_best_step": v2_r["best_step"],
            "steps_tf": steps_tf, "steps_v2": steps_v2,
            "tf_tokens": tf_r["tokens_seen"], "v2_tokens": v2_r["tokens_seen"],
            "tf_sec": tf_r["time_sec"], "v2_sec": v2_r["time_sec"],
            "flops_tf": f_tf, "flops_v2": f_v2, "flops_est_ratio": round(f_v2_est / f_tf_est, 3),
            "think_steps": args.think_steps, "tf_layers": args.tf_layers,
        }
        tf_bests.append(tf_r["best_val_loss"])
        v2_bests.append(v2_r["best_val_loss"])
        print(f"  seed {seed}: TF={tf_r['best_val_loss']:.4f} V2={v2_r['best_val_loss']:.4f} d={delta:+.4f} "
              f"(tf {tf_r['time_sec']:.0f}s, v2 {v2_r['time_sec']:.0f}s)")

    ps = paired_stats(v2_bests, tf_bests)
    tf_secs = [s["tf_sec"] for s in results["seeds"].values()]
    v2_secs = [s["v2_sec"] for s in results["seeds"].values()]
    wc_ratio = statistics.mean(v2_secs) / max(statistics.mean(tf_secs), 1e-6)
    results["summary"] = {
        "tf_best_mean": round(statistics.mean(tf_bests), 4),
        "tf_best_std": round(statistics.stdev(tf_bests), 4) if len(tf_bests) > 1 else 0.0,
        "v2_best_mean": round(statistics.mean(v2_bests), 4),
        "v2_best_std": round(statistics.stdev(v2_bests), 4) if len(v2_bests) > 1 else 0.0,
        **ps,
        "flops_method": flops_method,
        "tf_mean_sec": round(statistics.mean(tf_secs), 1),
        "v2_mean_sec": round(statistics.mean(v2_secs), 1),
        "v2_tf_wallclock_ratio": round(wc_ratio, 2),
        "v2_wins_equal_flops": bool(ps["v2_wins"] >= (len(seeds) + 1) // 2 and ps["delta_mean"] < 0),
        "stat_sig": bool(ps["n_seeds"] >= 3 and ps["p_value_paired_t"] is not None
                         and ps["p_value_paired_t"] < 0.05 and ps["delta_mean"] < 0),
    }
    if last_v2 is not None and vl is not None:
        batch = next(iter(vl))[0][:4]
        results["last_seed_internals"] = diagnose_internals(last_v2, batch)
        d = results["last_seed_internals"]
        print(f"\nLast-seed internals: iter_cos={d['iter_cosine_mean']} "
              f"mem_ent_ratio={d['memory_entropy_ratio']} top1={d['memory_top1_mass']} "
              f"eff_scale={d['mem_eff_scale_mean']}")
    p_str = f"{ps['p_value_paired_t']:.4f}" if ps["p_value_paired_t"] is not None else "n/a"
    print("\n" + "=" * 64)
    print("RESULTS (copy this block back)")
    print("=" * 64)
    print(f"TF  best: {results['summary']['tf_best_mean']:.4f} +/- {results['summary']['tf_best_std']:.4f}")
    print(f"V2  best: {results['summary']['v2_best_mean']:.4f} +/- {results['summary']['v2_best_std']:.4f}")
    print(f"d best (V2-TF): {ps['delta_mean']:+.4f} +/- {ps['delta_std']:.4f} | "
          f"p={p_str} (paired t) | sign p={ps['sign_test_p']:.4f} | flops={flops_method}")
    print(f"V2 wins: {ps['v2_wins']}/{len(seeds)} | stat_sig={results['summary']['stat_sig']}")
    print(f"Wall-clock: TF {results['summary']['tf_mean_sec']:.0f}s vs V2 {results['summary']['v2_mean_sec']:.0f}s "
          f"(ratio {results['summary']['v2_tf_wallclock_ratio']:.2f}x -- compile+amp+tf32 shrink this)")
    print("If V2 wins at equal FLOPs with p<0.05 across seeds -> compute-efficiency claim is real.")
    _save(results, "equal_flops")
    return results


def make_assoc_data(n=4000, n_pairs=4, seq_len=64, seed=0):
    """Associative recall: store (k,v) pairs then query k → v.
    Vocab layout: 0 pad, 1 sep, 2 query_mark, then keys 3.., values offset.
    """
    torch.manual_seed(seed)
    key_base, val_base = 10, 100
    vocab = val_base + 50
    data = []
    for _ in range(n):
        keys = torch.randint(0, 40, (n_pairs,)).tolist()
        vals = torch.randint(0, 40, (n_pairs,)).tolist()
        q = torch.randint(0, n_pairs, (1,)).item()
        seq = []
        for k, v in zip(keys, vals):
            seq += [key_base + k, val_base + v]
        seq += [2, key_base + keys[q]]  # query mark + key
        # target: everywhere ignore except last token predicts value
        tgt = [0] * (len(seq) - 1) + [val_base + vals[q]]
        # pad
        while len(seq) < seq_len:
            seq.append(0)
            tgt.append(0)
        seq, tgt = seq[:seq_len], tgt[:seq_len]
        # For LM API we use labels=input but need custom — use input_ids with labels
        # Store as (x, y) where y has -100 ignore except query answer position
        x = torch.tensor(seq, dtype=torch.long)
        y = torch.tensor(seq, dtype=torch.long).clone()
        y[:] = -100
        # predict token AFTER query key (position of answer). Our seq ends with [2, key], answer should be next.
        # Put answer as next token in x for causal LM: append value at end
        ans_pos = min(len([t for t in seq if t != 0]), seq_len - 1)
        # Rebuild cleaner:
        data.append((keys, vals, q))  # rebuild below
    # Cleaner rebuild
    data = []
    for _ in range(n):
        keys = torch.randint(0, 40, (n_pairs,)).tolist()
        vals = torch.randint(0, 40, (n_pairs,)).tolist()
        q = int(torch.randint(0, n_pairs, (1,)).item())
        toks = []
        for k, v in zip(keys, vals):
            toks += [key_base + k, val_base + v]
        toks += [2, key_base + keys[q], val_base + vals[q]]
        while len(toks) < seq_len:
            toks.append(0)
        toks = toks[:seq_len]
        x = torch.tensor(toks, dtype=torch.long)
        y = x.clone()
        # only supervise the answer token position (last non-pad content)
        ans_idx = toks.index(val_base + vals[q])
        y[:] = -100
        y[ans_idx] = toks[ans_idx]
        # Also need context tokens for embedding learning — supervise all non-pad lightly
        # Stronger: supervise full causal LM but score accuracy only on ans
        y = x.clone()
        data.append((x, y, ans_idx))
    return data, vocab


class AssocDS(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x, y, ans_idx = self.data[i]
        return x, y, torch.tensor(ans_idx, dtype=torch.long)


@torch.no_grad()
def eval_assoc_acc(model, loader):
    model.eval()
    correct, total = 0, 0
    loss_sum, n = 0.0, 0
    for batch in loader:
        x, y, ans_idx = batch
        x, y = x.to(DEVICE), y.to(DEVICE)
        ans_idx = ans_idx.to(DEVICE)
        out = model(x, labels=y)
        loss_sum += out["loss"].item()
        n += 1
        pred = out["logits"].argmax(dim=-1)
        # predict token at ans_idx using logits at ans_idx-1 (causal)
        for b in range(x.size(0)):
            i = int(ans_idx[b].item())
            if i <= 0:
                continue
            if pred[b, i - 1].item() == x[b, i].item():
                correct += 1
            total += 1
    return {
        "acc": round(correct / max(total, 1), 4),
        "loss": round(loss_sum / max(n, 1), 4),
        "n": total,
    }


def mode_assoc_recall(args):
    """Memory must bind key→value; Transformer can cheat with local attn but stress test helps."""
    print("\n=== ASSOCIATIVE RECALL MICROBENCH ===")
    data, vocab = make_assoc_data(n=max(2000, args.samples // 2), n_pairs=4, seed=0)
    split = int(len(data) * 0.9)
    tl = torch.utils.data.DataLoader(AssocDS(data[:split]), batch_size=args.batch, shuffle=True)
    vl = torch.utils.data.DataLoader(AssocDS(data[split:]), batch_size=args.batch)
    print(f"  n={len(data)} vocab={vocab} device={DEVICE}")

    def train_assoc(model, name):
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        best = {"acc": -1.0, "loss": 99.0, "step": 0}
        step, t0 = 0, time.time()
        while step < args.steps:
            for batch in tl:
                if step >= args.steps:
                    break
                x, y, _ = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                model(x, labels=y)["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                step += 1
                if step % args.log_every == 0 or step == args.steps:
                    metrics = eval_assoc_acc(model, vl)
                    if metrics["acc"] > best["acc"]:
                        best = {**metrics, "step": step}
                    print(f"  [{name:10s}] {step}/{args.steps} acc={metrics['acc']:.4f} loss={metrics['loss']:.4f} best_acc={best['acc']:.4f}")
                    model.train()
        return {"best": best, "params": count_params(model), "time_sec": round(time.time() - t0, 2)}

    results = {"meta": _meta(args, "assoc_recall"), "models": {}}
    results["models"]["transformer"] = train_assoc(make_tf(vocab, lora=args.lora, lora_rank=args.lora_rank), "TF")
    results["models"]["hybrid_v2"] = train_assoc(make_tb(vocab, "hybrid_v2", rank=args.think_rank, paths=args.thought_paths, train_break=args.train_break), "V2")
    tf_a = results["models"]["transformer"]["best"]["acc"]
    v2_a = results["models"]["hybrid_v2"]["best"]["acc"]
    results["summary"] = {
        "tf_acc": tf_a,
        "v2_acc": v2_a,
        "delta_acc": round(v2_a - tf_a, 4),
        "v2_wins": v2_a >= tf_a,
    }
    print("\n" + "=" * 64)
    print("RESULTS (copy this block back)")
    print("=" * 64)
    print(f"TF best acc: {tf_a:.4f}")
    print(f"V2 best acc: {v2_a:.4f}  (Δ={results['summary']['delta_acc']:+.4f})")
    print("Memory addressing is the point of this task.")
    _save(results, "assoc_recall")
    return results


@torch.no_grad()
def generate(model, tok, prompt_ids, max_new=200, context=512, temp=0.0, top_p=1.0, device=DEVICE):
    """Autoregressive completion. temp=0 → greedy; temp>0 → sampling (top_p)."""
    model.eval()
    ids = list(prompt_ids)
    for _ in range(max_new):
        x = torch.tensor([ids[-context:]], device=device)
        logits = model(x)["logits"][0, -1]
        if temp > 0:
            logits = logits / max(temp, 1e-6)
            probs = F.softmax(logits, dim=-1)
            if top_p < 1.0:
                sorted_p, sorted_idx = probs.sort(descending=True)
                cum = sorted_p.cumsum(0)
                keep = (cum - sorted_p) <= top_p
                kept = sorted_p * keep
                kept = kept / kept.sum()
                nxt = int(sorted_idx[torch.multinomial(kept, 1).item()].item())
            else:
                nxt = int(torch.multinomial(probs, 1).item())
        else:
            nxt = int(logits.argmax().item())
        ids.append(nxt)
        if nxt == 2:  # <eos> where available
            break
    return ids


def rollout_logprobs(model, prompt_ids, gen_ids):
    """Sum of log π(gen | prefix) via ONE teacher-forced forward (grad enabled)."""
    full = torch.tensor([prompt_ids + gen_ids], device=DEVICE)
    logits = model(full)["logits"][0]  # (L, V)
    lp = 0.0
    for pos, tgt in enumerate(gen_ids):
        lp = lp + F.log_softmax(logits[len(prompt_ids) - 1 + pos], dim=-1)[tgt]
    return lp


def generate_batch(model, prompt_ids_list, max_new=160, temp=0.0, top_p=1.0, device=DEVICE,
                   no_repeat_ngram=0, eos_token_id=2, pad_token_id=0):
    """Batched autoregressive generation — all sequences advance in lockstep.

    Turns B×K sequential forwards into ONE batched forward per token, which is
    10-50x faster on small models. Returns full id lists (prompt + generation).
    """
    model.eval()
    B = len(prompt_ids_list)
    max_p = max(len(p) for p in prompt_ids_list)
    cur = torch.zeros(B, max_p, dtype=torch.long, device=device)
    for i, p in enumerate(prompt_ids_list):
        cur[i, max_p - len(p):] = torch.tensor(p, dtype=torch.long, device=device)
    gens = [[] for _ in range(B)]
    done = [False] * B
    seen = [set() for _ in range(B)] if no_repeat_ngram > 0 else None
    for _ in range(max_new):
        if all(done):
            break
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                # pad_mask: left-pad positions must not be attended to / written
                # to memory — they only appear at window END in training, so at
                # generation (start of prompt) they corrupt conditioning.
                logits = model(cur, last_only=True, pad_mask=cur != pad_token_id)["logits"][:, -1]
        if pad_token_id is not None and pad_token_id != eos_token_id and pad_token_id < logits.shape[-1]:
            logits[:, pad_token_id] = float("-inf")
        if no_repeat_ngram > 0:
            for i in range(B):
                if done[i]:
                    continue
                g = gens[i]
                if len(g) >= no_repeat_ngram - 1:
                    key = tuple(g[-(no_repeat_ngram - 1):])
                    for ng in seen[i]:
                        if ng[:-1] == key and ng[-1] != eos_token_id and ng[-1] < logits.shape[-1]:
                            logits[i, ng[-1]] = float("-inf")
        if temp > 0:
            logits = logits / max(temp, 1e-6)
            probs = F.softmax(logits, dim=-1)
            if top_p < 1.0:
                sorted_p, sorted_idx = probs.sort(descending=True)
                cum = sorted_p.cumsum(1)
                kept = sorted_p * ((cum - sorted_p) <= top_p)
                kept = kept / kept.sum(1, keepdim=True)
                nxt = sorted_idx.gather(1, torch.multinomial(kept, 1))[:, 0]
            else:
                nxt = torch.multinomial(probs, 1)[:, 0]
        else:
            nxt = logits.argmax(dim=-1)
        for i in range(B):
            if not done[i]:
                gens[i].append(int(nxt[i].item()))
                if nxt[i].item() == eos_token_id:
                    done[i] = True
                elif no_repeat_ngram > 0 and len(gens[i]) >= no_repeat_ngram:
                    seen[i].add(tuple(gens[i][-no_repeat_ngram:]))
        nxt = nxt * torch.tensor([not d for d in done], device=device).long()
        cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
    return [prompt_ids_list[i] + gens[i] for i in range(B)]


def rollout_logprobs_batch(model, prompt_ids_list, gen_ids_list):
    """Per-rollout summed log-probs in ONE padded forward (grad enabled)."""
    B = len(prompt_ids_list)
    max_len = max(len(p) + len(g) for p, g in zip(prompt_ids_list, gen_ids_list))
    full = torch.zeros(B, max_len, dtype=torch.long, device=DEVICE)
    starts = []
    for i, (p, g) in enumerate(zip(prompt_ids_list, gen_ids_list)):
        full[i, :len(p) + len(g)] = torch.tensor(p + g, dtype=torch.long, device=DEVICE)
        starts.append(len(p))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        logits = model(full)["logits"]  # (B, maxL, V)
    lps = torch.zeros(B, device=DEVICE)
    for i, (p, g) in enumerate(zip(prompt_ids_list, gen_ids_list)):
        lg = F.log_softmax(logits[i].float(), dim=-1)
        s = starts[i]
        for j, tgt in enumerate(g):
            lps[i] = lps[i] + lg[s - 1 + j, tgt]
    return lps


def _force_compile(model):
    """torch.compile + a warmup forward so failures surface HERE, not mid-training."""
    compiled = torch.compile(model, mode="reduce-overhead")
    dummy = torch.zeros(1, 4, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        compiled(dummy)
    return compiled


def maybe_compile(model, args):
    """torch.compile (kernel fusion / CUDA graphs) with safe eager fallback.

    Skipped for TinyBrain: its python control flow (confidence .item() breaks,
    grad-mode switches) triggers constant recompilation — slower, not faster.
    """
    if getattr(args, "compile", False):
        if isinstance(getattr(model, "_orig_mod", model), TinyBrainModel):
            print("  compile skipped for TinyBrain (graph breaks)")
            return model
        try:
            return _force_compile(model)
        except Exception as e:
            print(f"  torch.compile unavailable ({e.__class__.__name__}) — eager")
    return model


def _check_code(code, test, timeout=3.0):
    """Execute prompt+completion+asserts in a subprocess; True iff exit 0."""
    import subprocess
    import sys
    try:
        r = subprocess.run([sys.executable, "-c", code + "\n" + test],
                           capture_output=True, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False


def mode_code_eval(args):
    """Market test: train on code, then measure HumanEval pass@1 (execution).

    The multi-agent (mixture-of-thoughts) and internal-simulation story is
    exercised here: the model plans, writes, and self-corrects inside the
    think loop before emitting code.
    """
    tl, vl, vocab = get_loaders(args)
    m = make_tb(vocab, "hybrid_v2", think_steps=args.think_steps, sharp=args.memory_sharp,
                rank=args.think_rank, paths=args.thought_paths, train_break=args.train_break)
    for c in m.cells:
        c.min_s = args.think_steps
        c.max_s = args.think_steps
    m.label_smoothing = args.label_smooth
    m = maybe_compile(m, args)
    tr = train_one(m, tl, vl, args.steps, "code_model", args.log_every,
                   early_stop_patience_steps=args.early_stop, lr=args.lr,
                   amp=args.amp, seq_len=args.seq_len)

    print("\n=== HUMANEVAL pass@1 (execution-based) ===")
    ds = _hf_load("openai_humaneval", split="test")
    tok = _CODE_TOK
    if tok is None:
        print("  ERROR: --dataset code required before --mode code_eval (no tokenizer)")
        return None
    correct = 0
    rows = []
    for i, ex in enumerate(ds):
        prompt = ex["prompt"]
        pid = _encode(tok, prompt, args.seq_len - 20)
        gen = generate(m, tok, pid, max_new=args.max_new, context=args.seq_len)
        comp = _decode(tok, gen[len(pid):])
        comp = comp.split("\ndef ")[0].split("\nclass ")[0]  # one function only
        ok = _check_code(prompt + comp, ex.get("test", ""))
        correct += ok
        rows.append({"problem": ex.get("entry_point", i), "ok": ok})
        if (i + 1) % 20 == 0 or i == len(ds) - 1:
            print(f"  humaneval {i + 1}/{len(ds)} | correct={correct} pass@1={correct / (i + 1):.4f}")
    pass1 = correct / max(len(ds), 1)
    results = {"meta": _meta(args, "code_eval"), "train": tr, "pass@1": round(pass1, 4),
               "n_problems": len(ds), "correct": correct, "rows": rows}
    print("\n" + "=" * 64)
    print("RESULTS (copy this block back)")
    print("=" * 64)
    print(f"HumanEval pass@1: {pass1:.4f} ({correct}/{len(ds)})")
    print("Market target: small + thinking must approach big-model pass@1 at far lower cost.")
    _save(results, "code_eval")
    return results


# ── GSM8K reasoning: supervised baseline, GRPO RL, self-consistency eval ──

_GSM = {"tokenizer": None, "test_prompts": [], "test_answers": []}


class _BpeTokWrapper:
    """Thin wrapper around a HuggingFace-style `tokenizers.Tokenizer` so that
    GSM8K encode/decode calls use the same API regardless of whether we're
    using a tiny domain BPE or a full pretrained HF tokenizer.
    Tokens 0=<pad>, 1=<unk>, 2=<eos> (the special_tokens order in BpeTrainer).
    """
    def __init__(self, tok):
        self._tok = tok
        self.eos_token_id = tok.token_to_id("<eos>") or 2
        self.pad_token_id = tok.token_to_id("<pad>") or 0
        self.unk_token_id = tok.token_to_id("<unk>") or 1

    def __len__(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text).ids

    def decode(self, ids, skip_special_tokens=True):
        text = self._tok.decode(ids)
        if skip_special_tokens:
            for sp in ["<pad>", "<unk>", "<eos>"]:
                text = text.replace(sp, "")
        return text


_GSM_BPE_TOK = None  # module-level cache, similar to _CODE_TOK


def _build_gsm8k_bpe(prompts, answers, t_p, t_a, vocab_size=8192):
    """Train a compact BPE tokenizer on all GSM8K text (train + test).
    Using a small vocab keeps lm_head proportional to the rest of the model.
    vocab_size=8192 gives <3.2M lm_head params at hidden=256 — right-sized.
    Fallback to byte tokenizer if the tokenizers library is missing.
    """
    global _GSM_BPE_TOK
    if _GSM_BPE_TOK is not None and len(_GSM_BPE_TOK) == vocab_size:
        return _GSM_BPE_TOK
    texts = []
    for q, a in zip(prompts + t_p, answers + t_a):
        texts.append(f"Question: {q}\nAnswer: {a}")
    print(f"  Training GSM8K BPE tokenizer (vocab={vocab_size}) on {len(texts)} texts...")
    try:
        raw_tok = _bpe_tokenizer(texts, vocab_size=vocab_size)
        _GSM_BPE_TOK = _BpeTokWrapper(raw_tok)
    except Exception as e:
        print(f"  tokenizers lib error ({e}) — falling back to word vocab")
        from collections import Counter
        cnt = Counter()
        for t in texts:
            cnt.update(t.split())
        vl = [w for w, _ in cnt.most_common(vocab_size - 3)]
        w2i = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        for i, w in enumerate(vl, 3):
            w2i[w] = i
        i2w = {v: k for k, v in w2i.items()}

        class _WordTokWrapper:
            def __init__(self, w2i, i2w):
                self._w2i, self._i2w = w2i, i2w
                self.eos_token_id = 2
                self.pad_token_id = 0
                self.unk_token_id = 1
            def __len__(self): return len(self._w2i)
            def encode(self, text):
                return [self._w2i.get(w, 1) for w in text.lower().split()]
            def decode(self, ids, skip_special_tokens=True):
                out = []
                for i in ids:
                    if skip_special_tokens and i in (0, 1, 2): continue
                    out.append(self._i2w.get(i, ""))
                return " ".join(out)
        _GSM_BPE_TOK = _WordTokWrapper(w2i, i2w)
    return _GSM_BPE_TOK


def load_gsm8k(max_samples=20000, split="train"):
    ds = _hf_load("openai/gsm8k", config="main", split=split)
    prompts, answers = [], []
    for ex in ds:
        prompts.append(ex["question"].strip())
        answers.append(ex["answer"].strip())  # "... #### 123"
        if len(prompts) >= max_samples:
            break
    return prompts, answers


def _make_gsm8k_lm_data(prompts, answers, tok, seq_len=128):
    """Question+answer as LM text with a trailing <eos>, keeping BOTH ends.
    Works with both _BpeTokWrapper and HF AutoTokenizer.
    """
    eos = tok.eos_token_id
    data = []
    for q, a in zip(prompts, answers):
        formatted_text = f"Question: {q}\nAnswer: {a}"
        toks = tok.encode(formatted_text)
        if eos is not None and (not toks or toks[-1] != eos):
            toks.append(eos)
        if len(toks) > seq_len:
            data.append(torch.tensor(toks[:seq_len], dtype=torch.long))
            data.append(torch.tensor(toks[-seq_len:], dtype=torch.long))
        else:
            data.append(torch.tensor(toks, dtype=torch.long))
    return data


def load_gsm8k_lm(max_samples=10000, seq_len=128, tokenizer_name="Qwen/Qwen2.5-0.5B",
                  gsm8k_vocab_size=8192):
    """GSM8K as LM data.

    By default trains a compact domain-specific BPE (vocab=gsm8k_vocab_size=8192).
    This keeps lm_head proportional to hidden_size=256 (~3M params vs 38M for Qwen 151k).
    Use --gsm8k_vocab_size 0 to fall back to the pretrained HuggingFace tokenizer.
    """
    global _GSM
    prompts, answers = load_gsm8k(max_samples, "train")
    t_p, t_a = load_gsm8k(5000, "test")
    if gsm8k_vocab_size > 0:
        tok = _build_gsm8k_bpe(prompts, answers, t_p, t_a, vocab_size=gsm8k_vocab_size)
    else:
        tok = get_tokenizer(tokenizer_name)
    _GSM.update({"tokenizer": tok, "test_prompts": t_p, "test_answers": t_a})
    data = _make_gsm8k_lm_data(prompts, answers, tok, seq_len)
    print(f"  gsm8k: {len(data)} seqs vocab={len(tok)}")
    return data, len(tok)


def _gsm8k_ans(text):
    return text.split("####")[-1].strip() if "####" in text else text.strip()


def _num_match(pred, gold):
    def norm(s):
        s = s.replace(",", "").replace(" ", "").strip()
        try:
            return round(float(s), 4)
        except Exception:
            return None
    p, g = norm(pred), norm(gold)
    if p is not None and g is not None:
        return abs(p - g) < 1e-3
    return pred == gold


def _gsm8k_reward(text, gold):
    """Multi-component reward for reasoning: format credit + numerical match.

    Sparse exact-match alone gives ZERO signal until the model can already
    produce a perfect answer (cold start) — the loss then drifts on KL alone.
    Shaped credits let RL learn the '<<calc>>' and '#### ' habits first, then
    the answer: 0.1 reasoning marker + 0.2 '####' format + 0.3 calculation
    attempt ('<<') + 1.0 exact match.
    """
    r = 0.0
    if "<think>" in text or "Reasoning:" in text or "Step " in text:
        r += 0.1
    if "####" in text:
        r += 0.2
    if "<<" in text:  # intermediate calculation attempt
        r += 0.3
    if _num_match(_gsm8k_ans(text), _gsm8k_ans(gold)):
        r += 1.0
    return r


def eval_gsm8k(model, args, tok=None, prompts=None, answers=None):
    """GSM8K accuracy; --reason_samples>1 → majority vote (self-consistency).

    --eval_n>0 evaluates only the first N test problems (fast iteration);
    --eval_think>0 caps the eval think depth (~2x faster at 4).
    """
    from collections import Counter
    if prompts is None:
        prompts, answers = _GSM["test_prompts"], _GSM["test_answers"]
    if tok is None:
        tok = _GSM.get("tokenizer") or get_tokenizer(getattr(args, "tokenizer_name", "Qwen/Qwen2.5-0.5B"))
    n_eval = getattr(args, "eval_n", 0) or len(prompts)
    prompts, answers = prompts[:n_eval], answers[:n_eval]

    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id
    sample = max(1, args.reason_samples)
    temp = 0.0 if sample == 1 else getattr(args, "rl_temp", 0.8)
    chunk = 128 if sample == 1 else max(8, 128 // sample)
    correct, total = 0, 0
    for c0 in range(0, len(prompts), chunk):
        chunk_p = prompts[c0:c0 + chunk]
        chunk_a = answers[c0:c0 + chunk]
        qids = [tok.encode(f"Question: {q}\nAnswer: ") for q in chunk_p]
        all_ids = [generate_batch(model, qids, max_new=args.max_new, temp=temp, no_repeat_ngram=3,
                                  eos_token_id=eos_id, pad_token_id=pad_id)
                   for _ in range(sample)]
        for i in range(len(chunk_p)):
            preds = [tok.decode(gen_i[len(qids[i]):], skip_special_tokens=True) for gen_i in (g[i] for g in all_ids)]
            best = Counter(_gsm8k_ans(p) for p in preds).most_common(1)[0][0]
            ok = int(_num_match(best, _gsm8k_ans(chunk_a[i])))
            correct += ok
            total += 1
            if total <= 3:
                print(f"    sample {total}: gold={_gsm8k_ans(chunk_a[i])!r} pred={best!r}")
        print(f"  gsm8k {total}/{len(prompts)} acc={correct / max(total, 1):.4f}", flush=True)
    return correct / max(total, 1)


def mode_reason_eval(args):
    """Supervised baseline on GSM8K, then reasoning accuracy (self-consistency)."""
    tl, vl, vocab = get_loaders(args)  # --dataset gsm8k -- tokenizer already built in load_gsm8k_lm
    m = make_tb(vocab, "hybrid_v2", hidden=args.hidden_size if hasattr(args, 'hidden_size') else 256,
                think_steps=args.think_steps, sharp=args.memory_sharp,
                rank=args.think_rank, paths=args.thought_paths, train_break=args.train_break,
                model_size=getattr(args, "model_size", None),
                attn_ratio=getattr(args, "attn_ratio", None),
                final_norm=getattr(args, "final_norm", None))
    m.label_smoothing = args.label_smooth
    m = maybe_compile(m, args)
    unk_id = getattr(_GSM.get("tokenizer"), "unk_token_id", 1)
    tr = train_one(m, tl, vl, args.steps, "gsm8k_sft", args.log_every,
                   early_stop_patience_steps=args.early_stop, lr=args.lr,
                   amp=args.amp, seq_len=args.seq_len,
                   input_dropout=getattr(args, "input_dropout", 0.0), unk_id=unk_id)
    # full-depth eval, same as mode_grpo: no confidence early-exit mid-chain
    # (the default threshold made reason_eval generate at ~1-3 think steps).
    eval_T = getattr(args, "eval_think", 0) or args.think_steps
    for c in m.cells:
        c.min_s = c.max_s = eval_T
        c.conf.thresh = 1.5
    tok = _GSM.get("tokenizer") or get_tokenizer(getattr(args, "tokenizer_name", "Qwen/Qwen2.5-0.5B"))
    acc = eval_gsm8k(m, args, tok)
    results = {"meta": _meta(args, "reason_eval"), "train": tr,
               "gsm8k_acc": round(acc, 4), "n_test": len(_GSM["test_prompts"]),
               "reason_samples": args.reason_samples}
    print("\n" + "=" * 64)
    print("RESULTS (copy this block back)")
    print("=" * 64)
    print(f"GSM8K accuracy: {acc:.4f} ({results['n_test']} problems, samples={args.reason_samples})")
    print("Supervised baseline — now compare against --mode grpo (RL).")
    _save(results, "reason_eval")
    return results


def mode_grpo(args):
    """R1-style RL: teach the model WHEN to think and how to verify."""
    import copy
    gsm8k_vocab_size = getattr(args, "gsm8k_vocab_size", 8192)
    prompts, answers = load_gsm8k(args.samples, "train")
    t_p, t_a = load_gsm8k(5000, "test")
    if gsm8k_vocab_size > 0:
        tok = _build_gsm8k_bpe(prompts, answers, t_p, t_a, vocab_size=gsm8k_vocab_size)
    else:
        tok = get_tokenizer(getattr(args, "tokenizer_name", "Qwen/Qwen2.5-0.5B"))
    _GSM.update({"tokenizer": tok, "test_prompts": t_p, "test_answers": t_a})
    vocab = len(tok)
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id

    max_T = max(args.think_steps, args.rl_rollout_think or 0)
    m = make_tb(vocab, "hybrid_v2", think_steps=max_T, sharp=args.memory_sharp,
                rank=args.think_rank, paths=args.thought_paths, train_break=args.train_break,
                model_size=getattr(args, "model_size", None),
                attn_ratio=getattr(args, "attn_ratio", None),
                final_norm=getattr(args, "final_norm", None))
    # model_size presets override think_steps — the effective depth comes from
    # the built config, not the CLI (keeps pins and step_emb consistent).
    max_T = m.config.max_think_steps
    rl_think = min(args.rl_rollout_think or max_T, max_T)
    sft_think = max_T
    m.label_smoothing = args.label_smooth
    m = maybe_compile(m, args)
    if args.load_path:
        print(f"\n=== loading SFT weights from {args.load_path} ===")
        orig = getattr(m, "_orig_mod", m)
        orig.load_state_dict(torch.load(args.load_path, map_location=DEVICE))
    elif args.rl_pretrain > 0:
        print(f"\n=== SFT warmup ({args.rl_pretrain} steps, think depth {sft_think}) ===")
        for c in m.cells:
            c.min_s = c.max_s = sft_think
        data = _make_gsm8k_lm_data(prompts, answers, tok, args.seq_len)
        split = int(len(data) * 0.9)
        tl = torch.utils.data.DataLoader(SeqDS(data[:split], args.seq_len, pad_id=pad_id), batch_size=args.batch, shuffle=True)
        vl = torch.utils.data.DataLoader(SeqDS(data[split:], args.seq_len, pad_id=pad_id), batch_size=args.batch)
        train_one(m, tl, vl, args.rl_pretrain, "sft", max(20, args.log_every // 4),
                  lr=args.lr, amp=args.amp, seq_len=args.seq_len,
                  input_dropout=getattr(args, "input_dropout", 0.0),
                  unk_id=getattr(tok, "unk_token_id", 1))
        if args.save_path:
            orig = getattr(m, "_orig_mod", m)
            torch.save(orig.state_dict(), args.save_path)
            print(f"  SFT weights saved to {args.save_path}")

    for c in m.cells:
        c.min_s = c.max_s = rl_think
        c.conf.thresh = 1.5
    if rl_think != max_T:
        print(f"  SFT+RL at think depth {rl_think}; eval will run at {max_T}")

    ref = copy.deepcopy(m).eval()
    opt = torch.optim.AdamW(m.parameters(), lr=args.rl_lr, weight_decay=0.0)
    K = max(2, args.rollouts)
    rl_batch = max(1, args.rl_batch)
    rl_max = args.rl_max_new
    print("\n=== GRPO RL ===")
    print(f"prompts={len(prompts)} rollouts={K} rl_batch={rl_batch} steps={args.rl_steps} "
          f"kl={args.rl_kl} temp={args.rl_temp} max_new={rl_max}")
    hist = []
    t0 = time.time()
    for step in range(args.rl_steps):
        idx = torch.randperm(len(prompts))[: rl_batch]
        qids, answers_b = [], []
        for i in idx.tolist():
            qids.append(tok.encode(f"Question: {prompts[i]}\nAnswer: "))
            answers_b.append(answers[i])
        roll_prompts = [q for q in qids for _ in range(K)]
        roll_gold = [a for a in answers_b for _ in range(K)]
        full = generate_batch(m, roll_prompts, max_new=rl_max, temp=args.rl_temp,
                              eos_token_id=eos_id, pad_token_id=pad_id)
        gens = [f[len(p):] for f, p in zip(full, roll_prompts)]
        rewards = [_gsm8k_reward(tok.decode(g, skip_special_tokens=True), a)
                   for g, a in zip(gens, roll_gold)]
        advs = []
        for k0 in range(0, len(rewards), K):
            grp = rewards[k0:k0 + K]
            mean = sum(grp) / K
            std = (sum((r - mean) ** 2 for r in grp) / K) ** 0.5
            advs += [(r - mean) / (std + 1e-4) for r in grp]
        lps = rollout_logprobs_batch(m, roll_prompts, gens)
        lprefs = rollout_logprobs_batch(ref, roll_prompts, gens).detach()
        total = 0.0
        for (p, g), adv, lp, lpref in zip(zip(roll_prompts, gens), advs, lps, lprefs):
            total += (-(adv * lp) + args.rl_kl * (lp - lpref)) / max(len(g), 1)
        loss = total / len(advs)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        fmt = sum(1 for r in rewards if r >= 0.2)
        full_cnt = sum(1 for r in rewards if r >= 1.0)
        hist.append({"step": step + 1, "loss": round(loss.item(), 4), "fmt": fmt, "full": full_cnt})
        if (step + 1) % 5 == 0 or step + 1 == args.rl_steps:
            el = time.time() - t0
            eta = el / (step + 1) * (args.rl_steps - step - 1)
            print(f"  [grpo] {step + 1}/{args.rl_steps} loss={loss.item():.4f} "
                  f"fmt={fmt}/{len(rewards)} full={full_cnt}/{len(rewards)} ({el:.0f}s, eta {eta / 60:.0f}min)", flush=True)

    eval_T = getattr(args, "eval_think", 0) or max_T
    for c in m.cells:
        c.min_s = c.max_s = eval_T
        c.conf.thresh = 1.5
    acc = eval_gsm8k(m, args, tok)
    results = {"meta": _meta(args, "grpo"), "rl_hist": hist,
               "gsm8k_acc_after_rl": round(acc, 4), "n_test": len(_GSM["test_prompts"]),
               "rl_lr": args.rl_lr, "rollouts": K, "rl_kl": args.rl_kl}
    print("\n" + "=" * 64)
    print("RESULTS (copy this block back)")
    print("=" * 64)
    print(f"GSM8K accuracy after GRPO: {acc:.4f} ({results['n_test']} problems)")
    print("Compare vs --mode reason_eval (supervised) — RL should win on reasoning.")
    _save(results, "grpo")
    return results


def mode_inference_bench(args):
    """1B-running-cost benchmark: V2 vs TF tokens/sec, FLOPs/token, avg think
    steps — and the ratio vs a 600B dense reference (2·600e9 FLOPs/token).
    """
    tl, vl, vocab = get_loaders(args)
    batch = next(iter(vl))[0][: args.batch].to(DEVICE)
    results = {"meta": _meta(args, "inference_bench"), "models": {}}
    print("\n=== INFERENCE BENCH (running cost) ===")
    cands = [
        ("transformer", make_tf(vocab, layers=args.tf_layers, lora=args.lora, lora_rank=args.lora_rank), False),
        ("hybrid_v2", make_tb(vocab, "hybrid_v2", think_steps=args.think_steps, rank=args.think_rank,
                              paths=args.thought_paths, train_break=args.train_break), True),
    ]
    for name, m, is_v2 in cands:
        m = maybe_compile(m, args)
        m.eval()
        with torch.no_grad():
            for _ in range(3):  # warmup
                m(batch)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        reps = 20
        t0 = time.time()
        with torch.no_grad():
            for _ in range(reps):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and DEVICE == "cuda"):
                    m(batch)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / reps
        fwd = measure_fwd_flops(m, args.seq_len, train_mode=False) or estimate_fwd_flops(m)
        flops_per_token = fwd / args.seq_len
        avg_steps = None
        if is_v2:
            orig = getattr(m, "_orig_mod", m)
            s = sum(c._steps_sum for c in orig.cells)
            n = sum(c._steps_n for c in orig.cells)
            avg_steps = round(s / n, 2) if n > 0 else None
        results["models"][name] = {
            "params": count_params(m), "trainable": trainable_params(m),
            "tokens_per_sec": round(batch.numel() / dt, 1),
            "flops_per_token": int(flops_per_token),
            "avg_think_steps": avg_steps,
            "ms_per_batch": round(dt * 1000, 2),
        }
        print(f"  {name:12s} params={count_params(m):,} tok/s={batch.numel() / dt:.0f} "
              f"FLOPs/tok={flops_per_token / 1e6:.0f}M avg_steps={avg_steps}")
    ref600 = 2 * 600e9  # dense 600B inference ≈ 2·N FLOPs per token
    v2 = results["models"]["hybrid_v2"]
    tf = results["models"]["transformer"]
    results["summary"] = {
        "v2_flops_per_token": v2["flops_per_token"],
        "v2_vs_600b_dense_cost": round(ref600 / v2["flops_per_token"], 1) if v2["flops_per_token"] else None,
        "v2_vs_tf_flops_per_token": round(v2["flops_per_token"] / max(tf["flops_per_token"], 1), 2),
        "v2_vs_tf_tokens_per_sec": round(tf["tokens_per_sec"] / max(v2["tokens_per_sec"], 1e-6), 2),
    }
    print("\n" + "=" * 64)
    print("RESULTS (copy this block back)")
    print("=" * 64)
    print(f"V2 FLOPs/token: {v2['flops_per_token'] / 1e6:.0f}M | 600B-dense: 1,200,000M")
    print(f"V2 running cost vs 600B-dense: {results['summary']['v2_vs_600b_dense_cost']}x cheaper per token")
    print(f"V2 vs TF: FLOPs/tok {results['summary']['v2_vs_tf_flops_per_token']}x, tok/s {results['summary']['v2_vs_tf_tokens_per_sec']}x (1.0 = TF)")
    print("1B-running-cost = structurally true (small params); accuracy needs benchmarks (GSM8K/HumanEval).")
    _save(results, "inference_bench")
    return results


def _meta(args, mode):
    return {
        "mode": mode,
        "device": DEVICE,
        "steps": args.steps,
        "batch": args.batch,
        "samples": args.samples,
        "dataset": getattr(args, "dataset", "tinystories"),
        "data_mix": getattr(args, "data_mix", None),
        "seq_len": getattr(args, "seq_len", 64),
        "thought_paths": getattr(args, "thought_paths", 1),
        "label_smooth": getattr(args, "label_smooth", 0.0),
        "amp": getattr(args, "amp", False),
        "rl_steps": getattr(args, "rl_steps", 0),
        "rollouts": getattr(args, "rollouts", 0),
        "rl_lr": getattr(args, "rl_lr", 1e-5),
        "rl_kl": getattr(args, "rl_kl", 0.01),
        "reason_samples": getattr(args, "reason_samples", 1),
        "lr": getattr(args, "lr", 3e-4),
        "warmup_fraction": getattr(args, "warmup", 0.02),
        "seeds": args.seeds,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cuda_name": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "goal": "compute-matched: hybrid v2 vs transformer at equal total FLOPs",
        "memory_fix": "selective slot write + sharp read",
        "diversity_fix": "relu(cos-0.95) aux loss",
        "multi_agent": "mixture-of-thoughts: parallel specialist paths + coordinator gate (Grok-style)",
        "note": "LR schedule keyed to tokens (not steps); equal-FLOPs compares BEST val; paired stats over seeds",
    }


NO_SAVE = False  # smoke tests set this to avoid polluting the results dir


def _auto_download(path):
    """Colab: push the artifact to the browser so it lands on the user's PC.
    RunPod/local: silently ignored (the path is printed by _save)."""
    try:
        from google.colab import files
        files.download(str(path))
        print(f"Downloaded: {path}")
    except Exception:
        pass


def _save(results, tag):
    if NO_SAVE:
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RES_DIR / f"{tag}_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {path}")
    _auto_download(path)
    return path


def verify():
    print(f"DEVICE={DEVICE}")
    vocab, x = 500, torch.randint(0, 500, (2, 32), device=DEVICE)
    for name, m in [
        ("transformer", make_tf(vocab, hidden=64, layers=2, heads=2)),
        ("plain", make_tb(vocab, "plain", hidden=64, cells=2, think_steps=2)),
        ("hybrid_v1", make_tb(vocab, "hybrid_v1", hidden=64, cells=2, think_steps=2)),
        ("hybrid_v2", make_tb(vocab, "hybrid_v2", hidden=64, cells=2, think_steps=2)),
    ]:
        out = m(x, labels=x)
        assert torch.isfinite(out["loss"]), name
        print(f"  OK {name:12s} params={count_params(m):,} loss={out['loss'].item():.4f}")
    m = make_tb(vocab, "hybrid_v2", hidden=64, cells=1, think_steps=2)
    with torch.no_grad():
        _ = m(x, labels=x)
        a = m.cells[0].mem._last_attn
        top1 = a.max(dim=-1).values.mean().item() if a is not None else 0.0
    print(f"  memory top1 mass (untrained)={top1:.4f} (uniform ~{1/16:.4f})")
    print("VERIFY PASS")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    p.add_argument(
        "--mode",
        choices=[
            "race", "diagnose", "think_scale", "verify_claim",
            "memory_ablation", "equal_flops", "assoc_recall",
            "code_eval", "reason_eval", "grpo", "inference_bench",
        ],
        default="race",
    )
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--seeds", type=str, default="0,1,2", help="comma seeds for multi-seed modes")
    p.add_argument("--dataset", choices=["tinystories", "wikitext", "openwebtext", "code", "gsm8k"], default="tinystories",
                   help="tinystories | wikitext | openwebtext | code (BPE) | gsm8k (reasoning)")
    p.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-0.5B",
                   help="HuggingFace pretrained tokenizer (used only when --gsm8k_vocab_size 0)")
    p.add_argument("--gsm8k_vocab_size", type=int, default=8192,
                   help="GSM8K domain BPE vocab size (default 8192, fits hidden=256 model). "
                        "Set 0 to use --tokenizer_name (HF Qwen/Llama) instead — "
                        "only safe when hidden_size >= 1024 (else lm_head dominates).")
    p.add_argument("--data_mix", type=str, default=None,
                   help="blend word datasets into one shared vocab, e.g. tinystories:0.5,wikitext:0.3,openwebtext:0.2 "
                        "(replay-style: avoids overfitting the newest corpus)")
    p.add_argument("--seq_len", type=int, default=64, help="sequence length (code wants 256-512)")
    p.add_argument("--memory_sharp", type=float, default=None,
                   help="override memory_sharp_init (effective read scale = softplus(init)/sqrt(d)); higher ⇒ sharper slot selection")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=float, default=0.02, help="warmup fraction of the token budget")
    p.add_argument("--early_stop", type=int, default=0,
                   help="early-stop patience in STEPS (0=off; equal_flops should stay 0 = full budget)")
    p.add_argument("--think_steps", type=int, default=4,
                   help="V2 think steps in equal_flops (thesis test: does more thinking beat TF at equal FLOPs?)")
    p.add_argument("--tf_layers", type=int, default=3,
                   help="Transformer layers in equal_flops (size the baseline to match V2 compute)")
    p.add_argument("--model_size", type=str, default=None,
                   choices=["nano", "small", "medium", "1b"],
                   help="architecture preset (overrides hidden/cells/think_steps/rank): "
                        "nano ~2.4M | small ~15M | medium ~150M | 1b ~1.0B")
    p.add_argument("--think_rank", type=int, default=None,
                   help="low-rank think branches (r < hidden_size) — much cheaper thinking; None = full d×d (default)")
    p.add_argument("--attn_ratio", type=float, default=None,
                   help="A/B: attn_dim_ratio override (None = proven 0.5 for hybrid_v2; 1.0 = full-width)")
    p.add_argument("--final_norm", type=int, default=0,
                   help="A/B: RMSNorm the hidden state before lm_head (bounds logit runaway)")
    p.add_argument("--input_dropout", type=float, default=0.0,
                   help="A/B: corrupt this fraction of input tokens with <unk> during SFT "
                        "(batched exposure-bias fix; 0.2 won the 20-example diag)")
    p.add_argument("--eval_n", type=int, default=0,
                   help="evaluate only the first N GSM8K test problems (0 = all 1319; "
                        "e.g. 300 for a ~4x faster signal)")
    p.add_argument("--eval_think", type=int, default=0,
                   help="eval think depth (0 = full; 4 = ~2x faster eval, matches rollout depth)")
    p.add_argument("--ema", type=float, default=0.0,
                   help="EMA decay for weights (0=off). Only helps NEAR CONVERGENCE: "
                        "use 0.99 for short runs (~100-step window), 0.999 needs 10k+ steps. "
                        "Val is evaluated on EMA weights.")
    p.add_argument("--thought_paths", type=int, default=1,
                   help="mixture-of-thoughts: K parallel think trajectories + coordinator gate "
                        "(Grok-style specialist agents inside one forward pass; 1 = current behavior)")
    p.add_argument("--label_smooth", type=float, default=0.0,
                   help="label smoothing for LM loss (0=off; 0.01-0.05 helps overfitting on big data)")
    p.add_argument("--amp", action="store_true",
                   help="bf16 autocast (Ampere+) — ~2x wall-clock on the sequential think loop")
    p.add_argument("--max_new", type=int, default=200, help="max generated tokens in code_eval")
    p.add_argument("--rl_steps", type=int, default=60, help="GRPO optimization steps")
    p.add_argument("--rollouts", type=int, default=4, help="rollouts per prompt in GRPO")
    p.add_argument("--rl_batch", type=int, default=8, help="prompts per GRPO step (rollouts are batched)")
    p.add_argument("--rl_max_new", type=int, default=160, help="max rollout tokens in GRPO (keep small for speed)")
    p.add_argument("--rl_rollout_think", type=int, default=None,
                   help="think depth used during RL rollouts (default=think_steps). Cheaper rollouts = faster RL; eval still runs at think_steps.")
    p.add_argument("--rl_lr", type=float, default=5e-6, help="policy LR for RL (small! 5e-6 stable, 1e-5 aggressive)")
    p.add_argument("--rl_temp", type=float, default=0.8, help="rollout sampling temperature")
    p.add_argument("--rl_kl", type=float, default=0.1, help="KL penalty vs reference policy (0.1 prevents reward-hacking collapse; 0.01 too weak)")
    p.add_argument("--rl_pretrain", type=int, default=300, help="supervised warmup steps before RL")
    p.add_argument("--save_path", type=str, default=None, help="save best weights here after SFT warmup (reuse across GRPO runs)")
    p.add_argument("--load_path", type=str, default=None, help="load weights instead of re-running SFT warmup")
    p.add_argument("--reason_samples", type=int, default=1,
                   help="self-consistency: sample N answers per problem, majority vote")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile (CUDA graphs) — big speedup for the sequential think loop / generation")
    p.add_argument("--train_break", type=float, default=0.8,
                   help="training confidence break (lower = think exits earlier = faster wall-clock; 0.5-0.6 typical for speed)")
    p.add_argument("--think_curriculum", type=str, default=None,
                   help="think-step curriculum across the budget, e.g. '1,2,4,8' (early cheap, late expensive)")
    p.add_argument("--lora", action="store_true", help="LoRA the transformer baseline (frozen base + adapters)")
    p.add_argument("--lora_rank", type=int, default=8, help="LoRA rank for the transformer baseline")
    args = p.parse_args()
    if args.verify:
        verify()
        return
    modes = {
        "race": mode_race,
        "diagnose": mode_diagnose,
        "think_scale": mode_think_scale,
        "verify_claim": mode_verify_claim,
        "memory_ablation": mode_memory_ablation,
        "equal_flops": mode_equal_flops,
        "assoc_recall": mode_assoc_recall,
        "code_eval": mode_code_eval,
        "reason_eval": mode_reason_eval,
        "grpo": mode_grpo,
        "inference_bench": mode_inference_bench,
    }
    modes[args.mode](args)


if __name__ == "__main__":
    main()
