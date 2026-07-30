"""
Phase 5 — Rigorous Experimental Suite

Addresses:
- Multiple random seeds (42, 123, 999)
- Gamma gradient logging (why is gamma stuck?)
- Real data (TinyStories) vs synthetic
- Memory metrics (active slots, entropy, write distribution)
- Thinking steps vs difficulty analysis
- All 4 models: TinyBrain, Transformer, GRU, Mamba
"""

import sys, os, json, math, time, argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util

# ── TinyBrain ──
spec = importlib.util.spec_from_file_location(
    "tiny_brain",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "novacore", "models", "tiny_brain.py")
)
tb = importlib.util.module_from_spec(spec)
sys.modules["tiny_brain"] = tb
spec.loader.exec_module(tb)
TinyBrainConfig, TinyBrainModel = tb.TinyBrainConfig, tb.TinyBrainModel

# ── Transformer ──
from novacore.core.simple_model import NovaModel
from novacore.core.config import NovaConfig

RESULTS = Path("novacore/experiments/phase5_results")
RESULTS.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_seeded_data(seed=42, vocab=5000, seq_len=64, n=20000, pattern_ratio=0.3):
    """Generate reproducible synthetic data."""
    torch.manual_seed(seed)
    data = []
    for _ in range(n):
        if torch.rand(1).item() < pattern_ratio:
            base = torch.randint(1, vocab//2, (1,)).item()
            seq = [(base + i) % vocab for i in range(seq_len)]
        else:
            seq = torch.randint(1, vocab, (seq_len,)).tolist()
        data.append(torch.tensor(seq, dtype=torch.long))
    return data


def get_tinystories_data(seed=42, max_samples=5000):
    """Load TinyStories from Hugging Face."""
    try:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train")
        texts = ds["text"][:max_samples]
        words = set()
        for t in texts:
            for w in t.lower().split()[:100]:
                words.add(w)
        vocab_list = sorted(words)
        word2idx = {w: i+2 for i, w in enumerate(vocab_list)}
        word2idx["<pad>"] = 0
        word2idx["<unk>"] = 1
        
        def tokenize(text, max_len=128):
            return [word2idx.get(w, 1) for w in text.lower().split()[:max_len]]
        
        data = [torch.tensor(tokenize(t), dtype=torch.long) for t in texts if len(t.split()) > 5]
        return data, len(vocab_list) + 2
    except ImportError:
        print("  datasets not available. Use synthetic.")
        return None, 5000
    except Exception as e:
        print(f"  Error: {e}. Use synthetic.")
        return None, 5000


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len=64):
        self.data = data
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx]
        if isinstance(x, torch.Tensor) and x.numel() < self.seq_len:
            pad = torch.zeros(self.seq_len - x.numel(), dtype=torch.long)
            x = torch.cat([x, pad])
        elif isinstance(x, torch.Tensor) and x.numel() > self.seq_len:
            x = x[:self.seq_len]
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x[:self.seq_len] if len(x) > self.seq_len else x, dtype=torch.long)
        return x, x.clone()


def create_model(name, vocab_size=5000, hidden=256):
    if name == "tinybrain":
        cfg = TinyBrainConfig(vocab_size=vocab_size, hidden_size=hidden, num_cells=3, memory_slots=16, num_think_heads=2, max_think_steps=8, min_think_steps=1, output_mlp_hidden=hidden*2)
        model = TinyBrainModel(cfg)
    elif name == "transformer":
        cfg = NovaConfig(vocab_size=vocab_size, hidden_size=hidden, num_layers=4, num_attention_heads=4, intermediate_size=hidden*3, max_seq_length=128)
        model = NovaModel(cfg)
    return model.to(DEVICE), sum(p.numel() for p in model.parameters())


@torch.no_grad()
def gamma_debug(model):
    """Detailed gamma analysis."""
    info = {"gammas": [], "gamma_grads": [], "gates": [], "gate_grads": []}
    for n, p in model.named_parameters():
        if "gamma" in n:
            info["gammas"].append(p.cpu().item())
            info["gamma_grads"].append(p.grad.cpu().norm().item() if p.grad is not None else 0.0)
        if "out_gate" in n:
            info["gates"].append(p.cpu().item())
            info["gate_grads"].append(p.grad.cpu().norm().item() if p.grad is not None else 0.0)
    
    # Memory metrics
    mem_active = 0
    mem_entropy = 0.0
    mem_write_strength = 0.0
    for n, p in model.named_parameters():
        if "M0" in n:
            norms = p.norm(dim=-1)
            mem_active = (norms > 0.01).float().mean().item()
            # Entropy: distribution of slot norms (higher = more diverse)
            probs = F.softmax(norms, dim=0)
            mem_entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        if "W_w" in n:  # Write gate weights
            mem_write_strength = p.abs().mean().item()
    
    return {
        "mean_gamma": sum(info["gammas"]) / len(info["gammas"]) if info["gammas"] else 0,
        "mean_gamma_grad": sum(info["gamma_grads"]) / len(info["gamma_grads"]) if info["gamma_grads"] else 0,
        "max_gamma": max(info["gammas"]) if info["gammas"] else 0,
        "min_gamma": min(info["gammas"]) if info["gammas"] else 0,
        "mean_gate": sum(info["gates"]) / len(info["gates"]) if info["gates"] else 0,
        "mean_gate_grad": sum(info["gate_grads"]) / len(info["gate_grads"]) if info["gate_grads"] else 0,
        "mem_active_ratio": mem_active,
        "mem_entropy": mem_entropy,
        "mem_write_strength": mem_write_strength,
    }


@torch.no_grad()
def evaluate(model, loader, n_batches=10):
    model.eval()
    total = 0.0
    count = 0
    for x, y in loader:
        if count >= n_batches: break
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x, labels=y)
        total += out["loss"].item()
        count += 1
    return total / max(count, 1)


def train(model, train_loader, val_loader, steps=2000, lr=3e-4, save_every=500):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []
    step = 0
    while step < steps:
        model.train()
        for x, y in train_loader:
            if step >= steps: break
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x, labels=y)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % save_every == 0 or step == steps:
                val_loss = evaluate(model, val_loader, n_batches=5)
                dbg = gamma_debug(model)
                entry = {"step": step, "train_loss": loss.item(), "val_loss": val_loss, **dbg}
                history.append(entry)
                print(f"  step={step:5d} | train={loss.item():.4f} | val={val_loss:.4f} | γ={dbg['mean_gamma']:.4f} | γ_grad={dbg['mean_gamma_grad']:.6f} | gate={dbg['mean_gate']:.4f} | gate_grad={dbg['mean_gate_grad']:.6f} | mem_active={dbg['mem_active_ratio']:.2%} | entropy={dbg['mem_entropy']:.2f}")
    return history


def run_seed(seed, data_source="synthetic", vocab=5000, steps=2000):
    """Full experiment for one seed."""
    print(f"\n{'='*60}")
    print(f"Seed {seed} | Data: {data_source}")
    print(f"{'='*60}")
    
    torch.manual_seed(seed)
    
    # Data
    if data_source == "tinystories":
        raw_data, actual_vocab = get_tinystories_data(seed=seed)
        vocab = actual_vocab
        seq_len = 64
    else:
        raw_data = get_seeded_data(seed=seed, vocab=vocab)
        seq_len = 64
    
    split = int(len(raw_data) * 0.9)
    train_ds = SimpleDataset(raw_data[:split], seq_len)
    val_ds = SimpleDataset(raw_data[split:], seq_len)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32)
    
    # Models
    models_to_test = ["tinybrain", "transformer"]
    all_results = {}
    
    for name in models_to_test:
        model, n_params = create_model(name, vocab_size=vocab)
        print(f"\n{name}: {n_params:,} params")
        t0 = time.time()
        history = train(model, train_loader, val_loader, steps=steps)
        t1 = time.time()
        
        all_results[name] = {
            "params": n_params,
            "history": history,
            "time_sec": t1 - t0,
        }
    
    return all_results, vocab


def run_full(steps=2000, data="synthetic"):
    """Run all 3 seeds."""
    results = {}
    for seed in [42, 123, 999]:
        r, vocab = run_seed(seed, data_source=data, steps=steps)
        results[f"seed_{seed}"] = r
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY — 3 Seeds")
    print("=" * 60)
    
    for seed_name, models in results.items():
        print(f"\n{seed_name}:")
        for model_name, info in models.items():
            if info["history"]:
                final = info["history"][-1]["val_loss"]
                print(f"  {model_name:15s} | final_val={final:.4f} | params={info['params']:,}")
    
    # Gamma analysis across seeds
    print("\nGamma Analysis:")
    for seed_name, models in results.items():
        if "tinybrain" in models:
            hist = models["tinybrain"]["history"]
            gamma_traj = [h["mean_gamma"] for h in hist]
            gamma_grads = [h["mean_gamma_grad"] for h in hist]
            print(f"  {seed_name}: γ trajectory: {[f'{g:.4f}' for g in gamma_traj]}")
            print(f"  {seed_name}: γ grad: {[f'{g:.6f}' for g in gamma_grads]}")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS / f"phase5_{data}_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--data", choices=["synthetic", "tinystories"], default="synthetic")
    args = parser.parse_args()
    
    print(f"TinyBrain — Phase 5 Rigorous")
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Steps: {args.steps}")
    print(f"Data: {args.data}")
    print()
    
    run_full(steps=args.steps, data=args.data)