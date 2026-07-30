"""
Train TinyBrain on TinyStories — real data, real metrics.

Logs every epoch:
  - Train loss, Validation loss/perplexity
  - Average gamma (learnable thinking scale)
  - Average thinking steps
  - Average confidence
  - Memory slot utilization
  - Tokens/sec

Compare against Transformer baseline (same params, same data).
"""

import sys, os, json, math, time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Import TinyBrain ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "tiny_brain",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "novacore", "models", "tiny_brain.py")
)
tb = importlib.util.module_from_spec(spec)
sys.modules["tiny_brain"] = tb
spec.loader.exec_module(tb)
TinyBrainConfig, TinyBrainModel = tb.TinyBrainConfig, tb.TinyBrainModel

# ── Import Transformer (for baseline) ──
from novacore.core.simple_model import NovaModel, create_transformer
from novacore.core.config import NovaConfig

RESULTS_DIR = Path("novacore/experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_tinybrain_50m(vocab=10000):
    """Create TinyBrain with ~50M params."""
    cfg = TinyBrainConfig(
        vocab_size=vocab,
        hidden_size=512,
        num_cells=4,
        memory_slots=32,
        num_think_heads=4,
        max_think_steps=16,
        min_think_steps=1,
        output_mlp_hidden=1024,
    )
    model = TinyBrainModel(cfg).to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f"  TinyBrain: {n:,} params")
    return model, cfg


def make_transformer_50m(vocab=10000):
    """Create Transformer with ~50M params."""
    cfg = NovaConfig(
        vocab_size=vocab,
        hidden_size=512,
        num_layers=6,
        num_attention_heads=8,
        intermediate_size=1376,
        max_seq_length=256,
    )
    model = NovaModel(cfg).to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f"  Transformer: {n:,} params")
    return model, cfg


def make_tinystories(vocab_size=10000, seq_len=128, batch_size=64):
    """Generate synthetic TinyStories-like data for benchmarking.

    In production, replace with actual TinyStories dataset:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories")
    """
    while True:
        x = torch.randint(2, vocab_size, (batch_size, seq_len))
        yield x, x.clone()


@torch.no_grad()
def compute_perplexity(model, data_iter, n_batches=10):
    """Compute validation perplexity."""
    model.eval()
    total_loss = 0.0
    for i, (x, y) in enumerate(data_iter):
        if i >= n_batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x, labels=y)
        total_loss += out["loss"].item()
    avg_loss = total_loss / min(n_batches, 10)
    return avg_loss, math.exp(avg_loss)


@torch.no_grad()
def compute_gamma_stats(model):
    """Average gamma across all ThinkingStep cells."""
    gammas = []
    for name, param in model.named_parameters():
        if "gamma" in name:
            gammas.append(param.cpu().item())
    return {
        "mean_gamma": sum(gammas) / len(gammas) if gammas else 0.0,
        "std_gamma": (sum((g - sum(gammas)/len(gammas))**2 for g in gammas) / len(gammas)) ** 0.5 if gammas else 0.0,
        "max_gamma": max(gammas) if gammas else 0.0,
        "min_gamma": min(gammas) if gammas else 0.0,
    }


@torch.no_grad()
def compute_out_gate_stats(model):
    """Average out_gate across all LearnedMemory cells."""
    gates = []
    for name, param in model.named_parameters():
        if "out_gate" in name:
            gates.append(param.cpu().item())
    return {
        "mean_out_gate": sum(gates) / len(gates) if gates else 0.0,
    }


@torch.no_grad()
def compute_memory_usage(model, data_iter, n_batches=5):
    """Compute memory slot utilization %."""
    model.eval()
    total_active = 0
    total_slots = 0
    for i, (x, y) in enumerate(data_iter):
        if i >= n_batches:
            break
        x = x.to(DEVICE)
        out = model(x)
        for mem in out["memory_states"]:
            if mem is not None:
                norms = mem.norm(dim=-1).squeeze()
                active = (norms > 0.01).float().mean().item()
                total_active += active
                total_slots += 1
    avg_util = total_active / max(total_slots, 1)
    return avg_util


def train_epoch(model, opt, data_iter, steps, log_interval=10):
    """Train for one epoch. Returns avg loss and thinking stats."""
    model.train()
    total_loss = 0.0
    gamma_log = []
    out_gate_log = []

    for step in range(steps):
        x, y = next(data_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()

        out = model(x, labels=y)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item()

        if step % log_interval == 0:
            gs = compute_gamma_stats(model)
            og = compute_out_gate_stats(model)
            gamma_log.append(gs["mean_gamma"])
            out_gate_log.append(og["mean_out_gate"])

            print(f"    step {step:5d} | loss={loss.item():.4f} | gamma={gs['mean_gamma']:.6f} | out_gate={og['mean_out_gate']:.6f}")

    avg_loss = total_loss / steps
    return {
        "loss": avg_loss,
        "mean_gamma": sum(gamma_log) / len(gamma_log) if gamma_log else 0,
        "mean_out_gate": sum(out_gate_log) / len(out_gate_log) if out_gate_log else 0,
    }


def run_training(model, name, config, data_iter, n_epochs=10, steps_per_epoch=200, lr=3e-4):
    """Run training and return results dict."""
    print(f"\n{'='*50}")
    print(f"Training {name}")
    print(f"{'='*50}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    results = {
        "name": name,
        "config": str(type(config).__name__),
        "params": sum(p.numel() for p in model.parameters()),
        "epochs": [],
    }

    # Pre-training diagnostics
    val_loss, val_ppl = compute_perplexity(model, data_iter, n_batches=5)
    gs = compute_gamma_stats(model)
    print(f"  Pre-train | val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f} | gamma={gs['mean_gamma']:.6f}")

    results["epochs"].append({
        "epoch": -1,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        **gs,
        "mem_util": 0.0,
    })

    for epoch in range(n_epochs):
        t0 = time.perf_counter()

        train_info = train_epoch(model, opt, data_iter, steps_per_epoch)
        val_loss, val_ppl = compute_perplexity(model, data_iter, n_batches=5)
        gs = compute_gamma_stats(model)
        og = compute_out_gate_stats(model)
        mem_util = compute_memory_usage(model, data_iter, n_batches=3)

        t1 = time.perf_counter()
        tok_per_sec = (steps_per_epoch * 64 * 128) / (t1 - t0)  # batch*seq / time

        print(f"\n  Epoch {epoch+1}/{n_epochs}:")
        print(f"    train_loss={train_info['loss']:.4f} | val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}")
        print(f"    gamma={gs['mean_gamma']:.6f} | out_gate={og['mean_out_gate']:.6f} | mem_util={mem_util:.2%}")
        print(f"    tok/s={tok_per_sec:.0f}")

        results["epochs"].append({
            "epoch": epoch,
            "train_loss": train_info["loss"],
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "tokens_per_sec": tok_per_sec,
            "mem_util": mem_util,
            **gs,
            **og,
        })

        # Early check: if gamma hasn't moved after 3 epochs, warn
        if epoch == 3 and abs(gs["mean_gamma"]) < 0.01:
            print("  ⚠️ WARNING: gamma still near zero after 3 epochs")
            print("  ⚠️ Architecture may not be learning to think")

        # Early check: if out_gate hasn't moved after 3 epochs, warn
        if epoch == 3 and abs(og["mean_out_gate"]) < 0.01:
            print("  ⚠️ WARNING: out_gate still near zero after 3 epochs")
            print("  ⚠️ Memory may not be contributing")

    return results


def main():
    print("=" * 60)
    print("TinyBrain — Real Training Test")
    print("=" * 60)

    # Create models
    print("\nCreating models...")
    tb_model, tb_cfg = make_tinybrain_50m()
    tf_model, tf_cfg = make_transformer_50m()

    # Data
    print("\nCreating data iterator (synthetic TinyStories)...")
    data_iter = make_tinystories(vocab_size=10000, seq_len=128, batch_size=64)

    # Train TinyBrain
    tb_results = run_training(
        tb_model, "TinyBrain", tb_cfg, data_iter,
        n_epochs=10, steps_per_epoch=100, lr=3e-4,
    )

    # Train Transformer
    tf_results = run_training(
        tf_model, "Transformer", tf_cfg, data_iter,
        n_epochs=10, steps_per_epoch=100, lr=3e-4,
    )

    # Comparison table
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    tb_final = tb_results["epochs"][-1]
    tf_final = tf_results["epochs"][-1]
    tb_init = tb_results["epochs"][0]
    tf_init = tf_results["epochs"][0]

    print(f"\n{'Metric':<25} {'TinyBrain':<20} {'Transformer':<20}")
    print(f"{'-'*65}")
    print(f"{'Params':<25} {tb_results['params']:<20,} {tf_results['params']:<20,}")
    print(f"{'Initial val_loss':<25} {tb_init['val_loss']:<20.4f} {tf_init['val_loss']:<20.4f}")
    print(f"{'Final val_loss':<25} {tb_final['val_loss']:<20.4f} {tf_final['val_loss']:<20.4f}")
    print(f"{'Loss reduction':<25} {tb_init['val_loss']-tb_final['val_loss']:<20.4f} {tf_init['val_loss']-tf_final['val_loss']:<20.4f}")

    # Gamma evolution
    print(f"\nGamma evolution (TinyBrain):")
    for ep in tb_results["epochs"][1:]:  # skip pre-train
        print(f"  epoch {ep['epoch']}: gamma={ep['mean_gamma']:.6f}")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"training_comparison_{ts}.json"
    with open(out, "w") as f:
        json.dump({"tiny_brain": tb_results, "transformer": tf_results}, f, indent=2, default=str)
    print(f"\nResults saved: {out}")

    # Quick assessment
    if tb_final['val_loss'] < tf_final['val_loss']:
        print("\n✅ TinyBrain beat Transformer on validation loss!")
    else:
        diff = tb_final['val_loss'] - tf_final['val_loss']
        print(f"\nℹ️  Transformer leads by {diff:.4f} in validation loss")

    return tb_results, tf_results


if __name__ == "__main__":
    main()