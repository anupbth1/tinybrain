"""
Scale Path — toward 1B TinyBrain ≈ 600B+ Transformer feel.

Thesis: quality should scale with *thinking compute*, not just params.
Path: close small-scale gap → prove think-step scaling → then grow to 1B.

Usage (Colab/RunPod — use ! in notebook cells):
  !python scale_path.py --verify
  !python scale_path.py --mode diagnose --steps 1000     # memory + diversity fix check
  !python scale_path.py --mode verify_claim --steps 2000 --seeds 0,1,2
  !python scale_path.py --mode think_scale --steps 800   # more steps ⇒ better?

Paste the RESULTS block back.
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
    return int(batch * (think + mem + attn))


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


def make_tb(vocab, variant="hybrid_v1", hidden=256, cells=3, think_steps=4):
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
        memory_sharp_init=5.0,
    )
    return TinyBrainModel(cfg).to(DEVICE)


def train_one(model, train_loader, val_loader, steps, name, log_every=200):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    hist, t0, step = [], time.time(), 0
    best_val, best_step = float("inf"), 0
    model.train()
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
            step += 1
            if step % log_every == 0 or step == steps:
                vl = eval_loss(model, val_loader)
                if vl < best_val:
                    best_val, best_step = vl, step
                gs = gate_stats(model) if not isinstance(model, NovaModel) else {}
                row = {
                    "step": step,
                    "val_loss": round(vl, 4),
                    "best_val_loss": round(best_val, 4),
                    "time_sec": round(time.time() - t0, 1),
                    **gs,
                }
                hist.append(row)
                extra = f" | γ={gs.get('gamma_mean', 0):.4f} gate={gs.get('out_gate_mean', 0):.4f}" if gs else ""
                print(f"  [{name:12s}] {step:4d}/{steps} | val={vl:.4f} best={best_val:.4f}{extra}")
                model.train()
    return {
        "name": name,
        "params": count_params(model),
        "approx_flops": estimate_fwd_flops(model),
        "final_val_loss": hist[-1]["val_loss"] if hist else round(eval_loss(model, val_loader), 4),
        "best_val_loss": round(best_val, 4),
        "best_step": best_step,
        "time_sec": round(time.time() - t0, 2),
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
    return {
        "iter_cosine_mean": round(sum(sims) / max(len(sims), 1), 4),
        "iter_cosine_list": [round(s, 4) for s in sims],
        "memory_entropy": round(ent, 4),
        "memory_entropy_ratio": round(ent / max(max_ent, 1e-6), 4),
        "memory_top1_mass": round(top1, 4),
        "out_norm": round(out.norm().item(), 4),
        "embed_norm": round(h.norm().item(), 4),
    }


def get_loaders(args):
    print("Loading TinyStories...")
    data, vocab = load_tinystories(args.samples)
    split = int(len(data) * 0.9)
    tl = torch.utils.data.DataLoader(SeqDS(data[:split]), batch_size=args.batch, shuffle=True)
    vl = torch.utils.data.DataLoader(SeqDS(data[split:]), batch_size=args.batch)
    print(f"  seqs={len(data)} vocab={vocab} device={DEVICE}")
    return tl, vl, vocab


def mode_race(args):
    """TF vs hybrid_v1 vs hybrid_v2 — compare BEST val (TF can overfit late)."""
    tl, vl, vocab = get_loaders(args)
    results = {"meta": _meta(args, "race"), "models": {}}
    print("\n=== RACE: Transformer vs Hybrid v1 vs Hybrid v2 ===")
    print("NOTE: report BEST val_loss (final can overfit, especially TF).")
    results["models"]["transformer"] = train_one(make_tf(vocab), tl, vl, args.steps, "Transformer", args.log_every)
    results["models"]["hybrid_v1"] = train_one(make_tb(vocab, "hybrid_v1"), tl, vl, args.steps, "hybrid_v1", args.log_every)
    results["models"]["hybrid_v2"] = train_one(make_tb(vocab, "hybrid_v2"), tl, vl, args.steps, "hybrid_v2", args.log_every)

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
        m = make_tb(vocab, variant)
        train_one(m, tl, vl, args.steps, variant, args.log_every)
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
        m = make_tb(vocab, "hybrid_v2", think_steps=tsteps)
        for c in m.cells:
            c.min_s = tsteps
            c.max_s = tsteps
        results["models"][name] = train_one(m, tl, vl, args.steps, name, args.log_every)
        results["models"][name]["think_steps"] = tsteps

    print("\n" + "=" * 64)
    print("RESULTS (copy this block back)")
    print("=" * 64)
    print(f"{'Config':12s} {'BestVal':>8s} {'Final':>8s} {'Params':>10s} {'~FLOPs':>12s}")
    losses = []
    for tsteps in [1, 2, 4, 8]:
        r = results["models"][f"v2_T{tsteps}"]
        losses.append(r["best_val_loss"])
        print(
            f"T={tsteps:<9d} {r['best_val_loss']:8.4f} {r['final_val_loss']:8.4f} "
            f"{r['params']:10,} {r['approx_flops']:12,}"
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
    import statistics as stats

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
        tl, vl, vocab = get_loaders(args)
        print(f"\n--- seed {seed} ---")
        tf_m = make_tf(vocab)
        v2_m = make_tb(vocab, "hybrid_v2")
        tf_r = train_one(tf_m, tl, vl, args.steps, f"TF_s{seed}", args.log_every)
        v2_r = train_one(v2_m, tl, vl, args.steps, f"V2_s{seed}", args.log_every)
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

    tf_mean, v2_mean = stats.mean(tf_bests), stats.mean(v2_bests)
    tf_std = stats.stdev(tf_bests) if len(tf_bests) > 1 else 0.0
    v2_std = stats.stdev(v2_bests) if len(v2_bests) > 1 else 0.0
    wins = sum(1 for a, b in zip(tf_bests, v2_bests) if b < a)
    results["summary"] = {
        "tf_best_mean": round(tf_mean, 4),
        "tf_best_std": round(tf_std, 4),
        "v2_best_mean": round(v2_mean, 4),
        "v2_best_std": round(v2_std, 4),
        "v2_wins": wins,
        "n_seeds": len(seeds),
        "claim_holds": bool(wins >= (len(seeds) + 1) // 2 and v2_mean < tf_mean),
    }
    print("\n" + "=" * 64)
    print("RESULTS (copy this block back)")
    print("=" * 64)
    print(f"TF  best: {tf_mean:.4f} ± {tf_std:.4f}")
    print(f"V2  best: {v2_mean:.4f} ± {v2_std:.4f}")
    print(f"V2 wins: {wins}/{len(seeds)} | claim_holds={results['summary']['claim_holds']}")
    print("If claim_holds → next: equal-FLOPs curve, then scale 50M→1B.")
    _save(results, "verify_claim")
    return results


def _meta(args, mode):
    return {
        "mode": mode,
        "device": DEVICE,
        "steps": args.steps,
        "batch": args.batch,
        "samples": args.samples,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cuda_name": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "goal": "1B TinyBrain compute ≈ 600B+ Transformer feel",
        "memory_fix": "selective slot write + sharp read",
        "diversity_fix": "relu(cos-0.95) aux loss",
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
        choices=["race", "diagnose", "think_scale", "verify_claim"],
        default="race",
    )
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--seeds", type=str, default="0,1,2", help="comma seeds for verify_claim")
    args = p.parse_args()
    if args.verify:
        verify()
        return
    if args.mode == "race":
        mode_race(args)
    elif args.mode == "diagnose":
        mode_diagnose(args)
    elif args.mode == "think_scale":
        mode_think_scale(args)
    else:
        mode_verify_claim(args)


if __name__ == "__main__":
    main()
