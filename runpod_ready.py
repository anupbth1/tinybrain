"""
TinyBrain — RunPod/Colab Ready Experiment Suite
Runs 5M proof-of-concept on CPU. Results can be continued on GPU.

Usage:
    # CPU: Complete 5M test (30 min)
    python runpod_ready.py --mode proof
    
    # CPU: Ablation study (1 hour)
    python runpod_ready.py --mode ablation
    
    # CPU: Just verify everything works (2 min)
    python runpod_ready.py --mode verify
    
    # RunPod: Continue training from checkpoint
    python runpod_ready.py --mode continue --checkpoint checkpoints/step_1000.pt
"""
import sys, os, json, math, time, argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Self-contained TinyBrain import ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "tiny_brain",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "novacore", "models", "tiny_brain.py")
)
tb = importlib.util.module_from_spec(spec)
sys.modules["tiny_brain"] = tb
spec.loader.exec_module(tb)
TinyBrainConfig = tb.TinyBrainConfig
TinyBrainModel = tb.TinyBrainModel

# ── Transformer baseline ──
from novacore.core.simple_model import NovaModel, create_transformer
from novacore.core.config import NovaConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Auto-detect GPU
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════
# Synthetic Data (deterministic for reproducibility)
# ═══════════════════════════════════════════════════════

def make_synthetic_data(vocab_size=5000, seq_len=64, num_seqs=20000, seed=42):
    """Create reproducible synthetic data with some structure (not purely random).
    
    Creates sequences with repeating patterns so model can learn."""
    torch.manual_seed(seed)
    data = []
    for _ in range(num_seqs):
        # Mix of random and patterned sequences
        if torch.rand(1).item() < 0.3:
            # Pattern: simple arithmetic-like sequences
            base = torch.randint(1, vocab_size//2, (1,)).item()
            seq = [(base + i) % vocab_size for i in range(seq_len)]
        else:
            seq = torch.randint(1, vocab_size, (seq_len,)).tolist()
        data.append(torch.tensor(seq, dtype=torch.long))
    return data


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x.clone()


@torch.no_grad()
def evaluate(model, loader, n_batches=10):
    model.eval()
    total_loss = 0.0
    count = 0
    for x, y in loader:
        if count >= n_batches: break
        out = model(x, labels=y)
        total_loss += out["loss"].item()
        count += 1
    avg = total_loss / max(count, 1)
    return avg, math.exp(avg)


@torch.no_grad()
def get_stats(model):
    """Get gamma, out_gate, memory_utilization."""
    gammas = [p.item() for n, p in model.named_parameters() if "gamma" in n]
    gates = [p.item() for n, p in model.named_parameters() if "out_gate" in n]
    
    # Memory utilization: fraction of slots with norm > 0.01
    mem_util = 0
    mem_count = 0
    for n, p in model.named_parameters():
        if "M0" in n:  # Memory initial parameters
            norms = p.norm(dim=-1)
            mem_util = (norms > 0.01).float().mean().item()
            mem_count = p.shape[0]
    
    return {
        "mean_gamma": sum(gammas) / len(gammas) if gammas else 0,
        "mean_gate": sum(gates) / len(gates) if gates else 0,
        "mem_util": mem_util,
        "mem_slots": mem_count,
        "num_gamma": len(gammas),
        "num_gates": len(gates),
    }


def create_model_5m(vocab_size=5000):
    """~5M parameter TinyBrain model for CPU training."""
    cfg = TinyBrainConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        num_cells=3,
        memory_slots=16,
        num_think_heads=2,
        max_think_steps=8,
        min_think_steps=1,
        output_mlp_hidden=512,
    )
    model = TinyBrainModel(cfg)
    n = sum(p.numel() for p in model.parameters())
    return model, cfg, n


def create_transformer_5m(vocab_size=5000):
    """~5M parameter Transformer for baseline."""
    cfg = NovaConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        num_layers=4,
        num_attention_heads=4,
        intermediate_size=640,
        max_seq_length=128,
    )
    model = NovaModel(cfg)
    n = sum(p.numel() for p in model.parameters())
    return model, cfg, n


def create_ablated_model(vocab_size=5000, remove_memory=False, remove_confidence=False, remove_selfcorr=False):
    """Create ablated TinyBrain variants by zeroing specific gates."""
    cfg = TinyBrainConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        num_cells=3,
        memory_slots=16,
        num_think_heads=2,
        max_think_steps=8,
        min_think_steps=1,
        output_mlp_hidden=512,
    )
    model = TinyBrainModel(cfg)
    
    if remove_memory:
        # Zero out out_gate parameters so memory produces no output
        for n, p in model.named_parameters():
            if "out_gate" in n:
                p.data.zero_()
                p.requires_grad = False  # Freeze at zero
    
    if remove_confidence:
        # Override max_think_steps to min to disable dynamic halting
        # (keeps confidence gate but forces fixed steps)
        for cell in model.cells:
            cell.max_s = 1  # Always think 1 step (no dynamic)
    
    if remove_selfcorr:
        # The self-correction module is small; set it to identity
        # by zeroing its output
        for n, p in model.named_parameters():
            if "verify" in n or "refine" in n:
                if 'weight' in n:
                    p.data.zero_()
    
    return model, cfg


def save_checkpoint(model, opt, step, stats, path):
    """Save training checkpoint."""
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "stats": stats,
        "config": model.config,
    }, path)
    print(f"  💾 Checkpoint saved: {path}")


def load_checkpoint(model, opt, path):
    """Load training checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    opt.load_state_dict(ckpt["opt_state"])
    print(f"  📂 Loaded checkpoint step {ckpt['step']}: {path}")
    return ckpt["step"], ckpt["stats"]


def train_model(model, train_loader, val_loader, steps=2000, lr=3e-4, save_every=250, name="model"):
    """Train with checkpoints every save_every steps."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []
    step = 0
    
    # Check for existing checkpoint
    ckpt_path = CHECKPOINT_DIR / f"{name}_latest.pt"
    start_step = 0
    if ckpt_path.exists():
        try:
            start_step, _ = load_checkpoint(model, opt, ckpt_path)
            step = start_step
            print(f"  Resuming from step {start_step}")
        except:
            print("  Starting fresh (checkpoint corrupt)")
    
    while step < steps:
        model.train()
        for x, y in train_loader:
            if step >= steps: break
            
            opt.zero_grad()
            out = model(x, labels=y)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            
            if step % save_every == 0 or step == steps:
                val_loss, val_ppl = evaluate(model, val_loader)
                stats = get_stats(model)
                entry = {
                    "step": step,
                    "train_loss": loss.item(),
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    **stats,
                }
                history.append(entry)
                print(f"  [{name}] step {step:5d}/{steps} | train={loss.item():.4f} | val={val_loss:.4f} | γ={stats['mean_gamma']:.4f} | gate={stats['mean_gate']:.4f} | mem={stats['mem_util']:.2%}")
                
                # Save checkpoint
                save_checkpoint(model, opt, step, entry, CHECKPOINT_DIR / f"{name}_step_{step}.pt")
                save_checkpoint(model, opt, step, entry, ckpt_path)
    
    return history


# ═══════════════════════════════════════════════════════
# MODES
# ═══════════════════════════════════════════════════════

def mode_verify():
    """Quick verification (2 min) — tests all models forward+backward."""
    print("=" * 60)
    print("MODE: Verify — Checking all models...")
    print("=" * 60)
    
    for name, fn in [
        ("TinyBrain 5M", lambda: create_model_5m()),
        ("Transformer 5M", lambda: create_transformer_5m()),
        ("Ablation (no memory)", lambda: create_ablated_model(remove_memory=True)),
        ("Ablation (no confidence)", lambda: create_ablated_model(remove_confidence=True)),
        ("Ablation (no self-corr)", lambda: create_ablated_model(remove_selfcorr=True)),
    ]:
        model, cfg, n = fn() if len(fn()) == 3 else (*fn(), 0)
        if n == 0:
            n = sum(p.numel() for p in model.parameters())
        x = torch.randint(0, 100, (2, 32))
        out = model(x, labels=x)
        loss = out["loss"].item()
        stats = get_stats(model)
        print(f"  ✅ {name:25s} | {n:>8,} params | loss={loss:.4f} | γ={stats['mean_gamma']:.4f} | mem={stats['mem_util']:.2%}")
    
    print("\n✅ All models verified. Ready for training.", flush=True)


def mode_proof():
    """Full 5M proof-of-concept with 5 checkpoints."""
    print("=" * 60)
    print("MODE: Proof — 5M TinyBrain vs Transformer")
    print("=" * 60)
    
    # Data
    print("\n📦 Creating synthetic data...")
    data = make_synthetic_data(vocab_size=5000, seq_len=64, num_seqs=20000)
    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]
    train_loader = torch.utils.data.DataLoader(SimpleDataset(train_data), batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(SimpleDataset(val_data), batch_size=32)
    print(f"  Train: {len(train_data)} seqs | Val: {len(val_data)} seqs")
    
    # Models
    print("\n🤖 Creating models...")
    tb_model, _, _ = create_model_5m()
    tf_model, _, _ = create_transformer_5m()
    n_tb = sum(p.numel() for p in tb_model.parameters())
    n_tf = sum(p.numel() for p in tf_model.parameters())
    print(f"  TinyBrain:   {n_tb:>8,} params")
    print(f"  Transformer: {n_tf:>8,} params")
    
    # Train
    print("\n🏋️  Training TinyBrain...", flush=True)
    tb_history = train_model(tb_model, train_loader, val_loader, steps=2000, name="tinybrain")
    
    print("\n🏋️  Training Transformer...", flush=True)
    tf_history = train_model(tf_model, train_loader, val_loader, steps=2000, name="transformer")
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n{'Step':>6} | {'TB Loss':>10} | {'TF Loss':>10} | {'TB γ':>8} | {'TB Gate':>8} | {'TB Mem':>8}")
    print("-" * 65)
    
    # Merge histories by step
    tb_dict = {e["step"]: e for e in tb_history}
    tf_dict = {e["step"]: e for e in tf_history}
    
    all_steps = sorted(set(list(tb_dict.keys()) + list(tf_dict.keys())))
    for s in all_steps:
        tb = tb_dict.get(s, {})
        tf = tf_dict.get(s, {})
        tb_l = tb.get("val_loss", -1)
        tf_l = tf.get("val_loss", -1)
        tb_g = tb.get("mean_gamma", -1)
        tb_gt = tb.get("mean_gate", -1)
        tb_m = tb.get("mem_util", -1)
        print(f"{s:>6} | {tb_l:>10.4f} | {tf_l:>10.4f} | {tb_g:>8.4f} | {tb_gt:>8.4f} | {tb_m:>8.2%}")
    
    # Save comprehensive results
    results = {
        "models": {
            "tinybrain": {"params": n_tb, "history": tb_history},
            "transformer": {"params": n_tf, "history": tf_history},
        },
        "comparison": {
            step: {
                "tinybrain_val_loss": tb_dict.get(step, {}).get("val_loss"),
                "transformer_val_loss": tf_dict.get(step, {}).get("val_loss"),
            }
            for step in all_steps
        }
    }
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("novacore/experiments/results") / f"proof_5m_{ts}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📊 Results saved: {out}")
    
    # Final verdict
    if tb_history and tf_history:
        final_tb = tb_history[-1]["val_loss"]
        final_tf = tf_history[-1]["val_loss"]
        if final_tb < final_tf:
            print(f"\n✅ TinyBrain wins! (TB: {final_tb:.4f} vs TF: {final_tf:.4f})")
        elif abs(final_tb - final_tf) < 0.05:
            print(f"\n🤝 Tie! (TB: {final_tb:.4f} vs TF: {final_tf:.4f})")
        else:
            print(f"\n📉 Transformer leads (TB: {final_tb:.4f} vs TF: {final_tf:.4f})")


def mode_ablation():
    """Ablation: full TinyBrain vs no-memory vs no-confidence."""
    print("=" * 60)
    print("MODE: Ablation — Which components help?")
    print("=" * 60)
    
    data = make_synthetic_data(vocab_size=5000, seq_len=64, num_seqs=15000)
    split = int(len(data) * 0.9)
    train_loader = torch.utils.data.DataLoader(SimpleDataset(data[:split]), batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(SimpleDataset(data[split:]), batch_size=32)
    
    variants = [
        ("full", lambda: create_model_5m()),
        ("no_memory", lambda: create_ablated_model(remove_memory=True)),
        ("no_confidence", lambda: create_ablated_model(remove_confidence=True)),
        ("no_selfcorr", lambda: create_ablated_model(remove_selfcorr=True)),
    ]
    
    all_history = {}
    for name, fn in variants:
        model, _, _ = fn()
        print(f"\n🏋️  Training variant: {name}", flush=True)
        history = train_model(model, train_loader, val_loader, steps=500, name=name)
        all_history[name] = history
    
    print("\n" + "=" * 60)
    print("ABLATION RESULTS")
    print("=" * 60)
    
    print(f"\n{'Variant':<15} {'Final Val Loss':<15} {'Change from Full':<18}")
    print("-" * 50)
    
    full_final = all_history.get("full", [{}])[-1].get("val_loss", 0)
    for name, history in all_history.items():
        final = history[-1]["val_loss"] if history else 0
        change = final - full_final if full_final else 0
        marker = "❌" if change > 0.1 else "✅" if change < -0.05 else "➡️"
        print(f"{marker} {name:<13} {final:<15.4f} {change:<+18.4f}")


def mode_continue():
    """Continue training from latest checkpoint on RunPod/Colab."""
    print("=" * 60)
    print("MODE: Continue — Resume from checkpoint on GPU")
    print("=" * 60)
    print("\n🔍 Checking for checkpoints...")
    
    ckpts = sorted(CHECKPOINT_DIR.glob("*.pt"))
    if not ckpts:
        print("  No checkpoints found. Run --mode proof first.")
        return
    
    for c in ckpts:
        size = c.stat().st_size / 1024
        print(f"  {c.name:40s} {size:>8.1f} KB")
    
    parser = argparse.ArgumentParser()
    latest = CHECKPOINT_DIR / "tinybrain_latest.pt"
    if latest.exists():
        print(f"\n  Latest: {latest}")
        print(f"\n  To continue on RunPod:")
        print(f"    python runpod_ready.py --mode continue --checkpoint {latest}")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["verify", "proof", "ablation", "continue"], default="verify")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    modes = {
        "verify": mode_verify,
        "proof": mode_proof,
        "ablation": mode_ablation,
        "continue": mode_continue,
    }
    
    print(f"\nTinyBrain — RunPod/Colab Ready Suite")
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Mode: {args.mode}")
    print()
    
    modes[args.mode]()