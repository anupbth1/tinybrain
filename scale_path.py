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

import torch
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


def _word_vocab(texts, top_k=30000, seq_len=64, max_words=50):
    """Word-level vocab (same style as load_tinystories: first max_words per text)."""
    words = set()
    for t in texts:
        for w in t.lower().split()[:max_words]:
            words.add(w)
    vl = sorted(words)[:top_k]
    w2i = {w: i + 2 for i, w in enumerate(vl)}
    w2i["<pad>"] = 0
    w2i["<unk>"] = 1
    return w2i


def _texts_to_data(texts, w2i, seq_len=64, min_words=5):
    data = []
    for t in texts:
        if len(t.split()) <= min_words:
            continue
        toks = [w2i.get(w, 1) for w in t.lower().split()[:seq_len]]
        data.append(torch.tensor(toks, dtype=torch.long))
    return data


def load_tinystories(max_samples=5000, seq_len=64):
    ds = _hf_load("roneneldan/TinyStories", split="train")
    texts = ds["text"][:max_samples]
    w2i = _word_vocab(texts)
    data = _texts_to_data(texts, w2i, seq_len)
    return data, len(w2i)


def load_wikitext(max_samples=20000, seq_len=64, top_k=20000):
    """WikiText-2 raw — out-of-distribution vs TinyStories (prose → encyclopedia).

    Note: HF moved the wikitext dataset to the Salesforce namespace; the bare
    repo id 'wikitext' fails on newer datasets versions (HfUriError).
    """
    ds = _hf_load("Salesforce/wikitext", config="wikitext-2-raw-v1", split="train")
    texts = [t.strip() for t in ds["text"] if len(t.strip()) > 5]
    if max_samples:
        texts = texts[:max_samples]
    w2i = _word_vocab(texts, top_k=top_k)
    data = _texts_to_data(texts, w2i, seq_len)
    print(f"  wikitext: {len(data)} seqs vocab={len(w2i)}")
    return data, len(w2i)


def load_openwebtext(max_samples=20000, seq_len=64, top_k=20000):
    """OpenWebText subset via streaming (downloads shards on the fly)."""
    ds = _hf_load("openwebtext", split="train", streaming=True)
    texts = []
    for ex in ds:
        t = ex["text"].strip()
        if len(t.split()) > 10:
            texts.append(t)
        if len(texts) >= max_samples:
            break
    w2i = _word_vocab(texts, top_k=top_k)
    data = _texts_to_data(texts, w2i, seq_len)
    print(f"  openwebtext: {len(data)} seqs vocab={len(w2i)}")
    return data, len(w2i)


class SeqDS(torch.utils.data.Dataset):
    def __init__(self, data, seq_len=64):
        self.data, self.seq_len = data, seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i]
        if x.numel() < self.seq_len:
            x = torch.cat([x, torch.zeros(self.seq_len - x.numel(), dtype=torch.long)])
        x = x[: self.seq_len]
        return x, x.clone()


def count_params(m):
    return sum(p.numel() for p in m.parameters())


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
def eval_loss(model, loader):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
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


def measure_fwd_flops(model, seq_len=SEQ_LEN):
    """Real per-step forward FLOPs via torch's FlopCounterMode (incl. lm_head).

    Hand-rolled counters have proven unreliable (they swung the TF:V2 ratio by
    >2x — measured TF is ~6x the old estimate). Measured numbers are what
    reviewers will trust. Returns None if the profiler is unavailable.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return None
    was_training = model.training
    model.train()  # training cost (dropout + full think loop), not eval cost
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
        model.train(was_training)


def make_tf(vocab, hidden=256, layers=3, heads=4):
    cfg = NovaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        num_layers=layers,
        num_attention_heads=heads,
        intermediate_size=hidden * 3,
        max_seq_length=64,
    )
    return NovaModel(cfg).to(DEVICE)


def make_tb(vocab, variant="hybrid_v1", hidden=256, cells=3, think_steps=4, sharp=None):
    """
    variants:
      plain      — no attention
      hybrid_v1  — post-cell lightweight attn
      hybrid_v2  — per-cell attn + step-conditioned + selective memory + diversity loss
    """
    if variant == "plain":
        use_post, every, ratio = False, False, 0.25
    elif variant == "hybrid_v1":
        use_post, every, ratio = True, False, 0.25
    elif variant == "hybrid_v2":
        use_post, every, ratio = False, True, 0.5
    else:
        raise ValueError(variant)
    cfg = TinyBrainConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        num_cells=cells,
        memory_slots=16,
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
    )
    return TinyBrainModel(cfg).to(DEVICE)


def train_one(model, train_loader, val_loader, steps, name, log_every=200,
              use_cosine=True, early_stop_patience_steps=0, lr=3e-4,
              warmup_fraction=0.02):
    """Train with an LR schedule keyed to TOKENS, not steps.

    Why tokens: in equal-FLOPs runs the two models get different step counts.
    Keying cosine + linear warmup to each model's own token budget means both
    models sit at the same LR phase for the same fraction of data seen — the
    step-count difference no longer biases the comparison.
    """
    tok_per_step = train_loader.batch_size * SEQ_LEN
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
            opt.zero_grad()
            loss = model(x, labels=y)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            step += 1
            lv = loss.item()
            ema_loss = lv if ema_loss is None else 0.9 * ema_loss + 0.1 * lv
            if step % log_every == 0 or step == steps:
                vl = eval_loss(model, val_loader)
                if vl < best_val - 1e-4:
                    best_val, best_step = vl, step
                    best_train_ema, lr_at_best = ema_loss, opt.param_groups[0]["lr"]
                gs = gate_stats(model) if not isinstance(model, NovaModel) else {}
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
                extra = f" | γ={gs.get('gamma_mean', 0):.4f} gate={gs.get('out_gate_mean', 0):.4f}" if gs else ""
                print(f"  [{name:12s}] {step:4d}/{steps} | val={vl:.4f} best={best_val:.4f} train={ema_loss:.4f}{extra}")
                model.train()
                if early_stop_patience_steps > 0 and step - best_step >= early_stop_patience_steps:
                    print(f"  [{name:12s}] early stop @ {step} (no improve for {step - best_step} steps)")
                    stopped_early = True
                    break
        if stopped_early or step >= steps:
            break
    wall = time.time() - t0
    return {
        "name": name,
        "params": count_params(model),
        "approx_flops": estimate_fwd_flops(model),
        "final_val_loss": hist[-1]["val_loss"] if hist else round(eval_loss(model, val_loader), 4),
        "best_val_loss": round(best_val, 4),
        "best_step": best_step,
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
        "gates": gate_stats(model) if not isinstance(model, NovaModel) else {},
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
    if args.dataset == "wikitext":
        data, vocab = load_wikitext(args.samples)
    elif args.dataset == "openwebtext":
        data, vocab = load_openwebtext(args.samples)
    else:
        data, vocab = load_tinystories(args.samples)
    split = int(len(data) * 0.9)
    g = torch.Generator().manual_seed(seed)  # deterministic shuffle per seed
    tl = torch.utils.data.DataLoader(SeqDS(data[:split]), batch_size=args.batch, shuffle=True, generator=g)
    vl = torch.utils.data.DataLoader(SeqDS(data[split:]), batch_size=args.batch)
    print(f"  seqs={len(data)} vocab={vocab} device={DEVICE}")
    return tl, vl, vocab


def mode_race(args):
    """TF vs hybrid_v1 vs hybrid_v2 — compare BEST val (TF can overfit late)."""
    tl, vl, vocab = get_loaders(args)
    results = {"meta": _meta(args, "race"), "models": {}}
    print("\n=== RACE: Transformer vs Hybrid v1 vs Hybrid v2 ===")
    print("NOTE: report BEST val_loss (final can overfit, especially TF).")
    results["models"]["transformer"] = train_one(make_tf(vocab), tl, vl, args.steps, "Transformer", args.log_every, lr=args.lr)
    results["models"]["hybrid_v1"] = train_one(make_tb(vocab, "hybrid_v1", sharp=args.memory_sharp), tl, vl, args.steps, "hybrid_v1", args.log_every, lr=args.lr)
    results["models"]["hybrid_v2"] = train_one(make_tb(vocab, "hybrid_v2", sharp=args.memory_sharp), tl, vl, args.steps, "hybrid_v2", args.log_every, lr=args.lr)

    tf_b = results["models"]["transformer"]["best_val_loss"]
    print("\n" + "=" * 64)
    print("RESULTS (copy this block back)")
    print("=" * 64)
    print(f"{'Model':14s} {'Params':>10s} {'BestVal':>8s} {'Final':>8s} {'Best@':>6s} {'ΔBestTF':>8s}")
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
        m = make_tb(vocab, variant, sharp=args.memory_sharp)
        train_one(m, tl, vl, args.steps, variant, args.log_every, lr=args.lr)
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
    print("If more steps ⇒ lower BEST loss, compute-scaling thesis is alive.")
    for tsteps in [1, 2, 4, 8]:
        name = f"v2_T{tsteps}"
        m = make_tb(vocab, "hybrid_v2", think_steps=tsteps, sharp=args.memory_sharp)
        for c in m.cells:
            c.min_s = tsteps
            c.max_s = tsteps
        results["models"][name] = train_one(m, tl, vl, args.steps, name, args.log_every, lr=args.lr)
        results["models"][name]["think_steps"] = tsteps
        mf = measure_fwd_flops(m)
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
        tf_m = make_tf(vocab)
        v2_m = make_tb(vocab, "hybrid_v2", sharp=args.memory_sharp)
        tf_r = train_one(tf_m, tl, vl, args.steps, f"TF_s{seed}", args.log_every,
                         early_stop_patience_steps=args.early_stop, lr=args.lr)
        v2_r = train_one(v2_m, tl, vl, args.steps, f"V2_s{seed}", args.log_every,
                         early_stop_patience_steps=args.early_stop, lr=args.lr)
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
        m = make_tb(vocab, "hybrid_v2", sharp=args.memory_sharp)
        train_one(m, tl, vl, args.steps, f"v2_s{seed}", args.log_every,
                  early_stop_patience_steps=args.log_every * 4, lr=args.lr)
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
        tf = make_tf(vocab, layers=args.tf_layers)
        v2 = make_tb(vocab, "hybrid_v2", think_steps=args.think_steps, sharp=args.memory_sharp)
        for c in v2.cells:
            c.min_s = args.think_steps
            c.max_s = args.think_steps
        # Prefer profiler-measured FLOPs (hand-rolled counters proved unreliable).
        f_tf = measure_fwd_flops(tf) or estimate_fwd_flops(tf)
        f_v2 = measure_fwd_flops(v2) or estimate_fwd_flops(v2)
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
                         early_stop_patience_steps=args.early_stop, lr=args.lr)
        v2_r = train_one(v2, tl, vl, steps_v2, f"V2_s{seed}", max(50, args.log_every // 2),
                         early_stop_patience_steps=args.early_stop, lr=args.lr)
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
        print(f"  seed {seed}: TF={tf_r['best_val_loss']:.4f} V2={v2_r['best_val_loss']:.4f} Δ={delta:+.4f} "
              f"(tf {tf_r['time_sec']:.0f}s, v2 {v2_r['time_sec']:.0f}s)")

    ps = paired_stats(v2_bests, tf_bests)
    results["summary"] = {
        "tf_best_mean": round(statistics.mean(tf_bests), 4),
        "tf_best_std": round(statistics.stdev(tf_bests), 4) if len(tf_bests) > 1 else 0.0,
        "v2_best_mean": round(statistics.mean(v2_bests), 4),
        "v2_best_std": round(statistics.stdev(v2_bests), 4) if len(v2_bests) > 1 else 0.0,
        **ps,
        "flops_method": flops_method,
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
    print(f"TF  best: {results['summary']['tf_best_mean']:.4f} ± {results['summary']['tf_best_std']:.4f}")
    print(f"V2  best: {results['summary']['v2_best_mean']:.4f} ± {results['summary']['v2_best_std']:.4f}")
    print(f"Δ best (V2-TF): {ps['delta_mean']:+.4f} ± {ps['delta_std']:.4f} | "
          f"p={p_str} (paired t) | sign p={ps['sign_test_p']:.4f} | flops={flops_method}")
    print(f"V2 wins: {ps['v2_wins']}/{len(seeds)} | stat_sig={results['summary']['stat_sig']}")
    print("If V2 wins at equal FLOPs with p<0.05 across seeds → compute-efficiency claim is real.")
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
    results["models"]["transformer"] = train_assoc(make_tf(vocab), "TF")
    results["models"]["hybrid_v2"] = train_assoc(make_tb(vocab, "hybrid_v2"), "V2")
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


def _meta(args, mode):
    return {
        "mode": mode,
        "device": DEVICE,
        "steps": args.steps,
        "batch": args.batch,
        "samples": args.samples,
        "dataset": getattr(args, "dataset", "tinystories"),
        "lr": getattr(args, "lr", 3e-4),
        "warmup_fraction": getattr(args, "warmup", 0.02),
        "seeds": args.seeds,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cuda_name": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "goal": "compute-matched: hybrid v2 vs transformer at equal total FLOPs",
        "memory_fix": "selective slot write + sharp read",
        "diversity_fix": "relu(cos-0.95) aux loss",
        "note": "LR schedule keyed to tokens (not steps); equal-FLOPs compares BEST val; paired stats over seeds",
    }


def _save(results, tag):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RES_DIR / f"{tag}_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {path}")


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
        ],
        default="race",
    )
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--seeds", type=str, default="0,1,2", help="comma seeds for multi-seed modes")
    p.add_argument("--dataset", choices=["tinystories", "wikitext", "openwebtext"], default="tinystories",
                   help="tinystories | wikitext (WikiText-2 raw) | openwebtext (streaming subset)")
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
    }
    modes[args.mode](args)


if __name__ == "__main__":
    main()
