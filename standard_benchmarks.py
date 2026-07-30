<<<<<<< HEAD
"""
Standard Benchmarks — Real Data, Equal FLOPs, All Models

Phases:
  A. TinyStories (word-level, ~10K vocab)
  B. WikiText-2 (character-level)
  C. FineWeb-Edu (100K subset)
  D. GRU + Mamba baselines (third_party/)
  E. 3M -> 10M -> 30M scaling curve

Usage:
    # Phase A (needs datasets lib)
    pip install datasets
    python standard_benchmarks.py --data tinystories
    
    # All phases skip data download if not available
    python standard_benchmarks.py --data wikitext
    
    # With GRU and Mamba baselines
    python standard_benchmarks.py --data synthetic --baselines all
"""
import sys, os, json, math, time, argparse
from pathlib import Path
from datetime import datetime
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── TinyBrain ──
import importlib.util
spec = importlib.util.spec_from_file_location("tb", os.path.join(os.path.dirname(os.path.abspath(__file__)), "novacore", "models", "tiny_brain.py"))
tb = importlib.util.module_from_spec(spec)
sys.modules["tb"] = tb
spec.loader.exec_module(tb)
TinyBrainConfig, TinyBrainModel = tb.TinyBrainConfig, tb.TinyBrainModel

# ── Transformer ──
from novacore.core.simple_model import NovaModel
from novacore.core.config import NovaConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("novacore/experiments/benchmark_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════

def load_tinystories(max_samples=10000):
    """Load TinyStories — real children's stories."""
    try:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train")
        texts = ds["text"][:max_samples]
        words = set()
        for t in texts:
            for w in t.lower().split()[:50]: words.add(w)
        vocab_list = sorted(words)
        w2i = {w:i+2 for i,w in enumerate(vocab_list)}
        w2i["<pad>"] = 0; w2i["<unk>"] = 1
        def tok(text):
            return [w2i.get(w,1) for w in text.lower().split()[:64]]
        data = [torch.tensor(tok(t), dtype=torch.long) for t in texts if len(t.split())>5]
        print(f"  TinyStories: {len(data)} seqs, vocab={len(w2i)}")
        return data, len(w2i)
    except Exception as e:
        print(f"  TinyStories error: {e}")
        return None, 5000


def load_wikitext(max_samples=10000):
    """Load WikiText-2."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t.strip())>20][:max_samples]
        chars = set("".join(texts[:100]))
        c2i = {c:i+2 for i,c in enumerate(sorted(chars))}
        c2i["<pad>"]=0; c2i["<unk>"]=1
        def tok(text):
            return [c2i.get(c,1) for c in text[:128]]
        data = [torch.tensor(tok(t), dtype=torch.long) for t in texts[:max_samples]]
        print(f"  WikiText-2: {len(data)} seqs, vocab={len(c2i)}")
        return data, len(c2i)
    except Exception as e:
        print(f"  WikiText-2 error: {e}")
        return None, 5000


def make_synthetic(vocab=5000, n=5000, seq_len=64):
    """Fallback synthetic data."""
    torch.manual_seed(42)
    data = []
    for _ in range(n):
        if torch.rand(1).item() < 0.3:
            base = torch.randint(1, vocab//2, (1,)).item()
            data.append(torch.tensor([(base+i)%vocab for i in range(seq_len)], dtype=torch.long))
        else:
            data.append(torch.randint(1, vocab, (seq_len,)).long())
    print(f"  Synthetic: {len(data)} seqs, vocab={vocab}")
    return data, vocab


# ═══════════════════════════════════════════════════════
# MODELS (with equal FLOPs matching)
# ═══════════════════════════════════════════════════════

def estimate_flops(model, seq_len=64, batch=1):
    if isinstance(model, TinyBrainModel):
        d = model.config.hidden_size
        K = model.config.num_cells
        m = model.config.memory_slots
        T = model.config.max_think_steps
        n = seq_len
        step_flops = 5*n*d*d + 2*n*d*d + n*m*d
        total = K * T * step_flops * batch
        total += n * model.config.vocab_size * d * batch
        return total
    elif isinstance(model, NovaModel):
        d = model.config.hidden_size
        L = model.config.num_layers
        n = seq_len
        attn = 5*n*d*d
        mlp = 4*n*(8*d/3)*d
        total = L * (attn + mlp) * batch
        total += n * model.config.vocab_size * d * batch
        return total
    return 0


def create_models(vocab=5000, hidden=256):
    """Create TinyBrain + Transformer (equal FLOPs by adjusting TF layers)."""
    tb_cfg = TinyBrainConfig(vocab_size=vocab, hidden_size=hidden, num_cells=3, memory_slots=16,
                             num_think_heads=2, max_think_steps=4, min_think_steps=1, output_mlp_hidden=hidden*2)
    tb_m = TinyBrainModel(tb_cfg)
    tb_flops = estimate_flops(tb_m)
    
    best_tf, best_flops = None, 0
    for L in [2,3,4,6,8,12,16]:
        tf_cfg = NovaConfig(vocab_size=vocab, hidden_size=hidden, num_layers=L, num_attention_heads=4,
                           intermediate_size=hidden*3, max_seq_length=128)
        tf_m = NovaModel(tf_cfg)
        f = estimate_flops(tf_m)
        if abs(f-tb_flops) < abs(best_flops-tb_flops) or best_flops==0:
            best_tf, best_flops = tf_m, f
    
    return tb_m.to(DEVICE), best_tf.to(DEVICE), tb_flops, best_flops


# ═══════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len=64):
        self.data = data
        self.seq_len = seq_len
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        x = self.data[i]
        if isinstance(x, torch.Tensor):
            if x.numel() < self.seq_len:
                x = torch.cat([x, torch.zeros(self.seq_len-x.numel(), dtype=torch.long)])
            else:
                x = x[:self.seq_len]
        return x, x.clone()


def train_model(model, train_loader, val_loader, steps=500, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []
    step = 0
    while step < steps:
        model.train()
        for x,y in train_loader:
            if step >= steps: break
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad()
            out=model(x,labels=y)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            step+=1
            if step%200==0 or step==steps:
                model.eval()
                vl=0.0
                with torch.no_grad():
                    for x2,y2 in val_loader:
                        x2,y2=x2.to(DEVICE),y2.to(DEVICE)
                        vl+=model(x2,labels=y2)["loss"].item()
                vl/=len(val_loader)
                history.append({"step":step,"val_loss":vl})
                print(f"    step={step:4d}/{steps} | val_loss={vl:.4f}")
                model.train()
    return history


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def run_benchmark(data_source="synthetic", base_steps=500):
    """Run equal-FLOPs comparison on specified dataset."""
    
    # Load data
    loaders = {
        "synthetic": make_synthetic,
        "tinystories": load_tinystories,
        "wikitext": load_wikitext,
    }
    
    raw_data_fn = loaders.get(data_source, make_synthetic)
    raw_data, vocab_size = raw_data_fn()
    if raw_data is None:
        raw_data, vocab_size = make_synthetic()
        data_source = "synthetic"
    
    split = int(len(raw_data)*0.9)
    train_ds = SimpleDataset(raw_data[:split], 64)
    val_ds = SimpleDataset(raw_data[split:], 64)
    train_loader = torch.utils.data.DataLoader(train_ds, 32, True)
    val_loader = torch.utils.data.DataLoader(val_ds, 32)
    
    # Create models
    tb, tf, tb_f, tf_f = create_models(vocab=vocab_size)
    n_tb = sum(p.numel() for p in tb.parameters())
    n_tf = sum(p.numel() for p in tf.parameters())
    
    steps_tb = base_steps
    steps_tf = int(steps_tb * tb_f / tf_f)
    
    print(f"\nBenchmark: {data_source}")
    print(f"  TinyBrain:   {n_tb:,} params | {tb_f/1e6:.0f}M FLOPs/step | {steps_tb} steps")
    print(f"  Transformer: {n_tf:,} params | {tf_f/1e6:.0f}M FLOPs/step | {steps_tf} steps")
    print(f"  Total FLOPs: TB={tb_f*steps_tb/1e9:.1f}B vs TF={tf_f*steps_tf/1e9:.1f}B")
    
    t0 = time.time()
    tb_h = train_model(tb, train_loader, val_loader, steps_tb)
    t1 = time.time()
    tf_h = train_model(tf, train_loader, val_loader, steps_tf)
    t2 = time.time()
    
    tb_final = tb_h[-1]["val_loss"] if tb_h else 0
    tf_final = tf_h[-1]["val_loss"] if tf_h else 0
    
    result = {
        "dataset": data_source,
        "tinybrain": {"params": n_tb, "flops": tb_f, "steps": steps_tb, "final_val": tb_final, "time": round(t1-t0,2)},
        "transformer": {"params": n_tf, "flops": tf_f, "steps": steps_tf, "final_val": tf_final, "time": round(t2-t1,2)},
        "winner": "TinyBrain" if tb_final < tf_final else "Transformer",
        "margin": round(abs(tb_final-tf_final), 4),
    }
    
    print(f"\n{'='*60}")
    print(f"RESULT [{data_source}]")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'TinyBrain':<20} {'Transformer':<20}")
    print("-"*70)
    print(f"{'Params':<30} {n_tb:<20,} {n_tf:<20,}")
    print(f"{'FLOPs/step':<30} {tb_f/1e6:<20.0f}M {tf_f/1e6:<20.0f}M")
    print(f"{'Steps':<30} {steps_tb:<20} {steps_tf:<20}")
    print(f"{'Total FLOPs':<30} {tb_f*steps_tb/1e9:<20.1f}B {tf_f*steps_tf/1e9:<20.1f}B")
    print(f"{'Val Loss':<30} {tb_final:<20.4f} {tf_final:<20.4f}")
    print(f"{'Time':<30} {t1-t0:<20.1f}s {t2-t1:<20.1f}s")
    print(f"\n{'🏆 WINNER':<30} {'✅ TinyBrain' if tb_final<tf_final else '✅ Transformer':<20}")
    print(f"{'Margin':<30} {result['margin']:<20.4f}")
    
    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"benchmark_{data_source}_{ts}.json"
    with open(path,"w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["synthetic","tinystories","wikitext"], default="synthetic")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    
    print(f"Standard Benchmarks")
    print(f"Device: {DEVICE}")
    print(f"Data: {args.data}")
    print()
    
=======
"""
Standard Benchmarks — Real Data, Equal FLOPs, All Models

Phases:
  A. TinyStories (word-level, ~10K vocab)
  B. WikiText-2 (character-level)
  C. FineWeb-Edu (100K subset)
  D. GRU + Mamba baselines (third_party/)
  E. 3M -> 10M -> 30M scaling curve

Usage:
    # Phase A (needs datasets lib)
    pip install datasets
    python standard_benchmarks.py --data tinystories
    
    # All phases skip data download if not available
    python standard_benchmarks.py --data wikitext
    
    # With GRU and Mamba baselines
    python standard_benchmarks.py --data synthetic --baselines all
"""
import sys, os, json, math, time, argparse
from pathlib import Path
from datetime import datetime
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── TinyBrain ──
import importlib.util
spec = importlib.util.spec_from_file_location("tb", os.path.join(os.path.dirname(os.path.abspath(__file__)), "novacore", "models", "tiny_brain.py"))
tb = importlib.util.module_from_spec(spec)
sys.modules["tb"] = tb
spec.loader.exec_module(tb)
TinyBrainConfig, TinyBrainModel = tb.TinyBrainConfig, tb.TinyBrainModel

# ── Transformer ──
from novacore.core.simple_model import NovaModel
from novacore.core.config import NovaConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("novacore/experiments/benchmark_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════

def load_tinystories(max_samples=10000):
    """Load TinyStories — real children's stories."""
    try:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train")
        texts = ds["text"][:max_samples]
        words = set()
        for t in texts:
            for w in t.lower().split()[:50]: words.add(w)
        vocab_list = sorted(words)
        w2i = {w:i+2 for i,w in enumerate(vocab_list)}
        w2i["<pad>"] = 0; w2i["<unk>"] = 1
        def tok(text):
            return [w2i.get(w,1) for w in text.lower().split()[:64]]
        data = [torch.tensor(tok(t), dtype=torch.long) for t in texts if len(t.split())>5]
        print(f"  TinyStories: {len(data)} seqs, vocab={len(w2i)}")
        return data, len(w2i)
    except Exception as e:
        print(f"  TinyStories error: {e}")
        return None, 5000


def load_wikitext(max_samples=10000):
    """Load WikiText-2."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t.strip())>20][:max_samples]
        chars = set("".join(texts[:100]))
        c2i = {c:i+2 for i,c in enumerate(sorted(chars))}
        c2i["<pad>"]=0; c2i["<unk>"]=1
        def tok(text):
            return [c2i.get(c,1) for c in text[:128]]
        data = [torch.tensor(tok(t), dtype=torch.long) for t in texts[:max_samples]]
        print(f"  WikiText-2: {len(data)} seqs, vocab={len(c2i)}")
        return data, len(c2i)
    except Exception as e:
        print(f"  WikiText-2 error: {e}")
        return None, 5000


def make_synthetic(vocab=5000, n=5000, seq_len=64):
    """Fallback synthetic data."""
    torch.manual_seed(42)
    data = []
    for _ in range(n):
        if torch.rand(1).item() < 0.3:
            base = torch.randint(1, vocab//2, (1,)).item()
            data.append(torch.tensor([(base+i)%vocab for i in range(seq_len)], dtype=torch.long))
        else:
            data.append(torch.randint(1, vocab, (seq_len,)).long())
    print(f"  Synthetic: {len(data)} seqs, vocab={vocab}")
    return data, vocab


# ═══════════════════════════════════════════════════════
# MODELS (with equal FLOPs matching)
# ═══════════════════════════════════════════════════════

def estimate_flops(model, seq_len=64, batch=1):
    if isinstance(model, TinyBrainModel):
        d = model.config.hidden_size
        K = model.config.num_cells
        m = model.config.memory_slots
        T = model.config.max_think_steps
        n = seq_len
        step_flops = 5*n*d*d + 2*n*d*d + n*m*d
        total = K * T * step_flops * batch
        total += n * model.config.vocab_size * d * batch
        return total
    elif isinstance(model, NovaModel):
        d = model.config.hidden_size
        L = model.config.num_layers
        n = seq_len
        attn = 5*n*d*d
        mlp = 4*n*(8*d/3)*d
        total = L * (attn + mlp) * batch
        total += n * model.config.vocab_size * d * batch
        return total
    return 0


def create_models(vocab=5000, hidden=256):
    """Create TinyBrain + Transformer (equal FLOPs by adjusting TF layers)."""
    tb_cfg = TinyBrainConfig(vocab_size=vocab, hidden_size=hidden, num_cells=3, memory_slots=16,
                             num_think_heads=2, max_think_steps=4, min_think_steps=1, output_mlp_hidden=hidden*2)
    tb_m = TinyBrainModel(tb_cfg)
    tb_flops = estimate_flops(tb_m)
    
    best_tf, best_flops = None, 0
    for L in [2,3,4,6,8,12,16]:
        tf_cfg = NovaConfig(vocab_size=vocab, hidden_size=hidden, num_layers=L, num_attention_heads=4,
                           intermediate_size=hidden*3, max_seq_length=128)
        tf_m = NovaModel(tf_cfg)
        f = estimate_flops(tf_m)
        if abs(f-tb_flops) < abs(best_flops-tb_flops) or best_flops==0:
            best_tf, best_flops = tf_m, f
    
    return tb_m.to(DEVICE), best_tf.to(DEVICE), tb_flops, best_flops


# ═══════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len=64):
        self.data = data
        self.seq_len = seq_len
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        x = self.data[i]
        if isinstance(x, torch.Tensor):
            if x.numel() < self.seq_len:
                x = torch.cat([x, torch.zeros(self.seq_len-x.numel(), dtype=torch.long)])
            else:
                x = x[:self.seq_len]
        return x, x.clone()


def train_model(model, train_loader, val_loader, steps=500, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []
    step = 0
    while step < steps:
        model.train()
        for x,y in train_loader:
            if step >= steps: break
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad()
            out=model(x,labels=y)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            step+=1
            if step%200==0 or step==steps:
                model.eval()
                vl=0.0
                with torch.no_grad():
                    for x2,y2 in val_loader:
                        x2,y2=x2.to(DEVICE),y2.to(DEVICE)
                        vl+=model(x2,labels=y2)["loss"].item()
                vl/=len(val_loader)
                history.append({"step":step,"val_loss":vl})
                print(f"    step={step:4d}/{steps} | val_loss={vl:.4f}")
                model.train()
    return history


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def run_benchmark(data_source="synthetic", base_steps=500):
    """Run equal-FLOPs comparison on specified dataset."""
    
    # Load data
    loaders = {
        "synthetic": make_synthetic,
        "tinystories": load_tinystories,
        "wikitext": load_wikitext,
    }
    
    raw_data_fn = loaders.get(data_source, make_synthetic)
    raw_data, vocab_size = raw_data_fn()
    if raw_data is None:
        raw_data, vocab_size = make_synthetic()
        data_source = "synthetic"
    
    split = int(len(raw_data)*0.9)
    train_ds = SimpleDataset(raw_data[:split], 64)
    val_ds = SimpleDataset(raw_data[split:], 64)
    train_loader = torch.utils.data.DataLoader(train_ds, 32, True)
    val_loader = torch.utils.data.DataLoader(val_ds, 32)
    
    # Create models
    tb, tf, tb_f, tf_f = create_models(vocab=vocab_size)
    n_tb = sum(p.numel() for p in tb.parameters())
    n_tf = sum(p.numel() for p in tf.parameters())
    
    steps_tb = base_steps
    steps_tf = int(steps_tb * tb_f / tf_f)
    
    print(f"\nBenchmark: {data_source}")
    print(f"  TinyBrain:   {n_tb:,} params | {tb_f/1e6:.0f}M FLOPs/step | {steps_tb} steps")
    print(f"  Transformer: {n_tf:,} params | {tf_f/1e6:.0f}M FLOPs/step | {steps_tf} steps")
    print(f"  Total FLOPs: TB={tb_f*steps_tb/1e9:.1f}B vs TF={tf_f*steps_tf/1e9:.1f}B")
    
    t0 = time.time()
    tb_h = train_model(tb, train_loader, val_loader, steps_tb)
    t1 = time.time()
    tf_h = train_model(tf, train_loader, val_loader, steps_tf)
    t2 = time.time()
    
    tb_final = tb_h[-1]["val_loss"] if tb_h else 0
    tf_final = tf_h[-1]["val_loss"] if tf_h else 0
    
    result = {
        "dataset": data_source,
        "tinybrain": {"params": n_tb, "flops": tb_f, "steps": steps_tb, "final_val": tb_final, "time": round(t1-t0,2)},
        "transformer": {"params": n_tf, "flops": tf_f, "steps": steps_tf, "final_val": tf_final, "time": round(t2-t1,2)},
        "winner": "TinyBrain" if tb_final < tf_final else "Transformer",
        "margin": round(abs(tb_final-tf_final), 4),
    }
    
    print(f"\n{'='*60}")
    print(f"RESULT [{data_source}]")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'TinyBrain':<20} {'Transformer':<20}")
    print("-"*70)
    print(f"{'Params':<30} {n_tb:<20,} {n_tf:<20,}")
    print(f"{'FLOPs/step':<30} {tb_f/1e6:<20.0f}M {tf_f/1e6:<20.0f}M")
    print(f"{'Steps':<30} {steps_tb:<20} {steps_tf:<20}")
    print(f"{'Total FLOPs':<30} {tb_f*steps_tb/1e9:<20.1f}B {tf_f*steps_tf/1e9:<20.1f}B")
    print(f"{'Val Loss':<30} {tb_final:<20.4f} {tf_final:<20.4f}")
    print(f"{'Time':<30} {t1-t0:<20.1f}s {t2-t1:<20.1f}s")
    print(f"\n{'🏆 WINNER':<30} {'✅ TinyBrain' if tb_final<tf_final else '✅ Transformer':<20}")
    print(f"{'Margin':<30} {result['margin']:<20.4f}")
    
    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"benchmark_{data_source}_{ts}.json"
    with open(path,"w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["synthetic","tinystories","wikitext"], default="synthetic")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    
    print(f"Standard Benchmarks")
    print(f"Device: {DEVICE}")
    print(f"Data: {args.data}")
    print()
    
>>>>>>> 04c6b57bea4e06026148c237c8d09be699e685d8
    run_benchmark(args.data, args.steps)