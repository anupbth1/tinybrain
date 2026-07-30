"""
5M Proof-of-Concept — CPU-trainable TinyStories training

Usage:
    # CPU (slower, but works)
    python -m novacore.training.train_5m --tiny
    
    # RunPod/Colab (with datasets library)
    python -m novacore.training.train_5m
    
    # Just test architecture (no data download)
    python -m novacore.training.train_5m --dry
"""
import sys, os, json, math, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Self-contained import
import importlib.util
spec = importlib.util.spec_from_file_location(
    "tiny_brain",
    os.path.join(os.path.dirname(__file__), "..", "models", "tiny_brain.py")
)
tb = importlib.util.module_from_spec(spec)
sys.modules["tiny_brain"] = tb
spec.loader.exec_module(tb)

TinyBrainConfig = tb.TinyBrainConfig
TinyBrainModel = tb.TinyBrainModel
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_5m_model(vocab_size: int = 10000) -> TinyBrainModel:
    """5M parameter model — trains in ~1 hour on CPU."""
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


def create_20m_model(vocab_size: int = 10000) -> TinyBrainModel:
    """20M parameter model."""
    cfg = TinyBrainConfig(
        vocab_size=vocab_size,
        hidden_size=384,
        num_cells=4,
        memory_slots=24,
        num_think_heads=4,
        max_think_steps=12,
        min_think_steps=1,
        output_mlp_hidden=768,
    )
    model = TinyBrainModel(cfg)
    n = sum(p.numel() for p in model.parameters())
    return model, cfg, n


def create_50m_model(vocab_size: int = 10000) -> TinyBrainModel:
    """50M parameter model — needs GPU."""
    cfg = TinyBrainConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        num_cells=6,
        memory_slots=32,
        num_think_heads=4,
        max_think_steps=16,
        min_think_steps=1,
        output_mlp_hidden=1024,
    )
    model = TinyBrainModel(cfg)
    n = sum(p.numel() for p in model.parameters())
    return model, cfg, n


def get_tinystories(token_limit: int = 50000):
    """Load TinyStories dataset. Falls back to synthetic if datasets not installed."""
    try:
        from datasets import load_dataset
        print("Loading TinyStories from Hugging Face...")
        ds = load_dataset("roneneldan/TinyStories", split="train")
        texts = ds["text"][:token_limit]
        print(f"  Loaded {len(texts)} stories")
        
        # Build simple tokenizer (word-level for 5M test)
        words = set()
        for t in texts:
            for w in t.lower().split()[:100]:
                words.add(w)
        vocab = {w: i+2 for i, w in enumerate(sorted(words)[:9998])}
        vocab["<pad>"] = 0
        vocab["<unk>"] = 1
        
        def tokenize(text):
            return [vocab.get(w, 1) for w in text.lower().split()[:256]]
        
        data = [tokenize(t) for t in texts]
        data = [d for d in data if len(d) > 10]
        print(f"  Tokenized to {len(data)} sequences, vocab={len(vocab)}")
        return data, len(vocab)
        
    except ImportError:
        print("datasets library not available. Using synthetic data.")
        return None, 10000
    except Exception as e:
        print(f"Error loading TinyStories: {e}")
        return None, 10000


class TinyStoriesDataset(torch.utils.data.Dataset):
    """Simple TinyStories dataset."""
    def __init__(self, data, seq_len=128):
        self.data = data
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data) * max(1, len(self.data[0]) // self.seq_len) if self.data else 1000
        
    def __getitem__(self, idx):
        if self.data is None:
            # Synthetic data fallback
            x = torch.randint(2, 10000, (self.seq_len,))
            return x, x.clone()
        
        story_idx = idx % len(self.data)
        story = self.data[story_idx]
        if len(story) > self.seq_len:
            start = (idx // len(self.data)) % (len(story) - self.seq_len)
            chunk = story[start:start + self.seq_len]
        else:
            chunk = story + [0] * (self.seq_len - len(story))
        x = torch.tensor(chunk[:self.seq_len], dtype=torch.long)
        return x, x.clone()


@torch.no_grad()
def evaluate(model, loader, n_batches=5):
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        if count >= n_batches: break
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x, labels=y)
        total += out["loss"].item()
        count += 1
    return total / max(count, 1)


def train(model, train_loader, val_loader, steps=1000, lr=3e-4, log_interval=100):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []
    
    for step in range(steps):
        model.train()
        x, y = next(iter(train_loader))
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        opt.zero_grad()
        out = model(x, labels=y)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if step % log_interval == 0:
            val_loss = evaluate(model, val_loader)
            gammas = [p.item() for n, p in model.named_parameters() if "gamma" in n]
            gates = [p.item() for n, p in model.named_parameters() if "out_gate" in n]
            
            entry = {
                "step": step,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "mean_gamma": sum(gammas) / len(gammas) if gammas else 0,
                "mean_gate": sum(gates) / len(gates) if gates else 0,
            }
            history.append(entry)
            
            print(f"  step {step:5d} | train={loss.item():.4f} | val={val_loss:.4f} | γ={entry['mean_gamma']:.4f} | gate={entry['mean_gate']:.4f}")
    
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true", help="5M tiny model (CPU)")
    parser.add_argument("--medium", action="store_true", help="20M model")
    parser.add_argument("--large", action="store_true", help="50M model (needs GPU)")
    parser.add_argument("--dry", action="store_true", help="Test without data")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    
    print("=" * 60)
    print("TinyBrain — 5M Proof-of-Concept")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # Create model
    if args.tiny or args.dry:
        model, cfg, n = create_5m_model()
    elif args.medium:
        model, cfg, n = create_20m_model()
    elif args.large:
        model, cfg, n = create_50m_model()
    else:
        model, cfg, n = create_5m_model()
    
    model = model.to(DEVICE)
    print(f"\nModel: {n:,} params (hidden={cfg.hidden_size}, cells={cfg.num_cells}, mem={cfg.memory_slots})")
    
    if args.dry:
        print("Dry run — testing forward pass only")
        x = torch.randint(0, 1000, (2, 64)).to(DEVICE)
        out = model(x, labels=x)
        print(f"  Forward OK. Loss: {out['loss'].item():.4f}")
        print(f"  Memory states: {[m.shape if m is not None else None for m in out['memory_states']]}")
        return
    
    # Get data
    print("\nLoading data...")
    data, vocab_size = get_tinystories()
    if data is None:
        print("  Using synthetic data (no datasets library)")
    else:
        cfg.vocab_size = vocab_size
        # Recreate model with correct vocab
        model = create_model_from_cfg(cfg).to(DEVICE)
    
    # Create dataloaders
    seq_len = 128
    dataset = TinyStoriesDataset(data, seq_len)
    split = int(len(dataset) * 0.9)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [split, len(dataset) - split])
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16)
    
    print(f"  Train: {len(train_ds)} sequences")
    print(f"  Val:   {len(val_ds)} sequences")
    
    # Train
    print(f"\nTraining ({args.steps} steps)...")
    t0 = time.time()
    history = train(model, train_loader, val_loader, steps=args.steps, lr=args.lr)
    t1 = time.time()
    
    print(f"\nTraining complete in {t1-t0:.1f}s")
    
    # Results
    if history:
        final = history[-1]
        print(f"\nFinal: train_loss={final['train_loss']:.4f}, val_loss={final['val_loss']:.4f}")
        print(f"       gamma={final['mean_gamma']:.4f}, gate={final['mean_gate']:.4f}")
    
    # Save
    results_dir = Path(__file__).parent.parent / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"5m_proof_{int(time.time())}.json"
    with open(out, "w") as f:
        json.dump({
            "params": n,
            "history": history,
            "config": str(cfg),
            "device": DEVICE,
        }, f, indent=2, default=str)
    print(f"\nResults saved: {out}")


def create_model_from_cfg(cfg):
    return TinyBrainModel(cfg)


if __name__ == "__main__":
    main()