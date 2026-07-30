"""
Hybrid TinyBrain vs Transformer — RunPod / Colab critical test.

Goal: close the TinyStories gap (TB 4.71 vs TF 4.30) by adding
lightweight causal attention after thinking cells.

Usage (GPU recommended):
  pip install torch datasets
  python hybrid_compare.py                  # TinyStories, 500 steps
  python hybrid_compare.py --steps 1000
  python hybrid_compare.py --synthetic      # no HF download
  python hybrid_compare.py --verify         # 10s smoke test

Paste the printed RESULTS table back after the run.
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

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
RES_DIR = Path("novacore/experiments/hybrid_results")
RES_DIR.mkdir(parents=True, exist_ok=True)


def load_tinystories(max_samples=5000, seq_len=64):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train")
    texts = ds["text"][:max_samples]
    words = set()
    for t in texts:
        for w in t.lower().split()[:50]:
            words.add(w)
    vl = sorted(words)
    w2i = {w: i + 2 for i, w in enumerate(vl)}
    w2i["<pad>"] = 0
    w2i["<unk>"] = 1

    def tok(t):
        return [w2i.get(w, 1) for w in t.lower().split()[:seq_len]]

    data = [torch.tensor(tok(t), dtype=torch.long) for t in texts if len(t.split()) > 5]
    return data, len(w2i)


def load_synthetic(vocab=2000, seq_len=64, n=4000, seed=42):
    torch.manual_seed(seed)
    data = []
    for _ in range(n):
        if torch.rand(1).item() < 0.4:
            base = torch.randint(2, vocab // 2, (1,)).item()
            seq = [(base + i) % vocab for i in range(seq_len)]
        else:
            seq = torch.randint(2, vocab, (seq_len,)).tolist()
        data.append(torch.tensor(seq, dtype=torch.long))
    return data, vocab


class SeqDS(torch.utils.data.Dataset):
    def __init__(self, data, seq_len=64):
        self.data = data
        self.seq_len = seq_len

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
        "gamma_mean": round(sum(gammas) / len(gammas), 4) if gammas else 0.0,
        "out_gate_mean": round(sum(gates) / len(gates), 4) if gates else 0.0,
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


def train_one(model, train_loader, val_loader, steps, name, log_every=100):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    hist = []
    t0 = time.time()
    step = 0
    model.train()
    while step < steps:
        for x, y in train_loader:
            if step >= steps:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x, labels=y)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % log_every == 0 or step == steps:
                vl = eval_loss(model, val_loader)
                gs = gate_stats(model) if any("gamma" in n for n, _ in model.named_parameters()) else {}
                row = {"step": step, "val_loss": round(vl, 4), "time_sec": round(time.time() - t0, 1), **gs}
                hist.append(row)
                extra = f" | γ={gs.get('gamma_mean', 0):.4f} gate={gs.get('out_gate_mean', 0):.4f}" if gs else ""
                print(f"  [{name:18s}] step {step:4d}/{steps} | val_loss={vl:.4f}{extra}")
                model.train()
    final = hist[-1] if hist else {"val_loss": eval_loss(model, val_loader)}
    return {
        "name": name,
        "params": count_params(model),
        "final_val_loss": final["val_loss"],
        "time_sec": round(time.time() - t0, 2),
        "history": hist,
        "gates": gate_stats(model) if any("gamma" in n for n, _ in model.named_parameters()) else {},
    }


def make_tb(vocab, use_attn, hidden=256, cells=3, steps=4):
    cfg = TinyBrainConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        num_cells=cells,
        memory_slots=16,
        num_think_heads=2,
        max_think_steps=steps,
        min_think_steps=1,
        output_mlp_hidden=hidden * 2,
        use_token_attn=use_attn,
        attn_dim_ratio=0.25,
        attn_heads=1,
        gamma_init=0.1,
        out_gate_init=0.1,
    )
    return TinyBrainModel(cfg).to(DEVICE)


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


def verify():
    print(f"DEVICE={DEVICE}")
    vocab = 500
    x = torch.randint(0, vocab, (2, 32), device=DEVICE)
    for name, m in [
        ("transformer", make_tf(vocab, hidden=64, layers=2, heads=2)),
        ("tinybrain_plain", make_tb(vocab, use_attn=False, hidden=64, cells=2, steps=2)),
        ("tinybrain_hybrid", make_tb(vocab, use_attn=True, hidden=64, cells=2, steps=2)),
    ]:
        out = m(x, labels=x)
        assert torch.isfinite(out["loss"]), name
        print(f"  OK {name:20s} params={count_params(m):,} loss={out['loss'].item():.4f}")
    print("VERIFY PASS")


def run(args):
    print("=" * 64)
    print("HYBRID COMPARE — Transformer vs TinyBrain vs Hybrid")
    print(f"DEVICE={DEVICE} | steps={args.steps} | data={'synthetic' if args.synthetic else 'tinystories'}")
    print("=" * 64)

    if args.synthetic:
        data, vocab = load_synthetic()
    else:
        print("Loading TinyStories...")
        data, vocab = load_tinystories(args.samples)
    split = int(len(data) * 0.9)
    tl = torch.utils.data.DataLoader(SeqDS(data[:split]), batch_size=args.batch, shuffle=True)
    vl = torch.utils.data.DataLoader(SeqDS(data[split:]), batch_size=args.batch)
    print(f"  seqs={len(data)} vocab={vocab} batch={args.batch}")

    results = {
        "meta": {
            "device": DEVICE,
            "steps": args.steps,
            "batch": args.batch,
            "vocab": vocab,
            "n_seqs": len(data),
            "data": "synthetic" if args.synthetic else "tinystories",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "cuda_name": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        },
        "models": {},
    }

    print("\n--- Transformer ---")
    results["models"]["transformer"] = train_one(
        make_tf(vocab), tl, vl, args.steps, "Transformer", args.log_every
    )

    print("\n--- TinyBrain (no attn) ---")
    results["models"]["tinybrain_plain"] = train_one(
        make_tb(vocab, use_attn=False), tl, vl, args.steps, "TB-plain", args.log_every
    )

    print("\n--- TinyBrain Hybrid ---")
    results["models"]["tinybrain_hybrid"] = train_one(
        make_tb(vocab, use_attn=True), tl, vl, args.steps, "TB-hybrid", args.log_every
    )

    # Summary table
    tf_l = results["models"]["transformer"]["final_val_loss"]
    plain_l = results["models"]["tinybrain_plain"]["final_val_loss"]
    hyb_l = results["models"]["tinybrain_hybrid"]["final_val_loss"]
    results["summary"] = {
        "transformer_val_loss": tf_l,
        "plain_val_loss": plain_l,
        "hybrid_val_loss": hyb_l,
        "hybrid_vs_transformer": round(hyb_l - tf_l, 4),
        "hybrid_vs_plain": round(hyb_l - plain_l, 4),
        "hybrid_beats_transformer": hyb_l < tf_l,
        "hybrid_beats_plain": hyb_l < plain_l,
    }

    print("\n" + "=" * 64)
    print("RESULTS (copy this block back)")
    print("=" * 64)
    print(f"{'Model':22s} {'Params':>10s} {'ValLoss':>8s} {'Δ vs TF':>8s} {'Time':>7s}")
    for key, label in [
        ("transformer", "Transformer"),
        ("tinybrain_plain", "TB-plain"),
        ("tinybrain_hybrid", "TB-hybrid"),
    ]:
        r = results["models"][key]
        d = r["final_val_loss"] - tf_l
        print(f"{label:22s} {r['params']:10,} {r['final_val_loss']:8.4f} {d:+8.4f} {r['time_sec']:6.1f}s")
    s = results["summary"]
    print("-" * 64)
    print(f"Hybrid vs Transformer: {s['hybrid_vs_transformer']:+.4f}  "
          f"({'WIN' if s['hybrid_beats_transformer'] else 'still behind'})")
    print(f"Hybrid vs Plain:       {s['hybrid_vs_plain']:+.4f}  "
          f"({'helps' if s['hybrid_beats_plain'] else 'no help'})")
    if "gates" in results["models"]["tinybrain_hybrid"]:
        g = results["models"]["tinybrain_hybrid"]["gates"]
        print(f"Hybrid gates: γ={g.get('gamma_mean')}  out_gate={g.get('out_gate_mean')}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RES_DIR / f"hybrid_{ts}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=100)
    args = p.parse_args()
    if args.verify:
        verify()
    else:
        run(args)


if __name__ == "__main__":
    main()
