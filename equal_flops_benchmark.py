<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
"""
Equal-FLOPs Benchmark — Fair Comparison

Goal: Compare TinyBrain vs Transformer at SAME compute budget.
Not same steps. Not same params. SAME FLOPS.

Method:
1. Measure FLOPs for 1 Transformer step
2. Count how many TinyBrain steps fit in same FLOPs
3. Compare loss at equal FLOPs budget
"""
import sys, os, math, time, json
from pathlib import Path
from datetime import datetime
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("tb", os.path.join(os.path.dirname(os.path.abspath(__file__)), "novacore", "models", "tiny_brain.py"))
tb = importlib.util.module_from_spec(spec)
sys.modules["tb"] = tb
spec.loader.exec_module(tb)
TinyBrainConfig, TinyBrainModel = tb.TinyBrainConfig, tb.TinyBrainModel

from novacore.core.simple_model import NovaModel
from novacore.core.config import NovaConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("novacore/experiments/equal_flops_results")
OUT.mkdir(parents=True, exist_ok=True)


def estimate_flops_per_step(model, seq_len=64, batch=1):
    """Estimate FLOPs per forward pass."""
    if isinstance(model, TinyBrainModel):
        d = model.config.hidden_size
        K = model.config.num_cells
        m = model.config.memory_slots
        T = model.config.max_think_steps
        n = seq_len
        # Each thinking step: ~5·n·d² (thinking) + 2·n·d² (memory project) + n·m·d (memory read)
        step_flops = 5 * n * d * d + 2 * n * d * d + n * m * d
        total = K * T * step_flops * batch
        # Add embed + output
        total += n * model.config.vocab_size * d * batch  # lm_head
        return total
    elif isinstance(model, NovaModel):
        d = model.config.hidden_size
        L = model.config.num_layers
        n = seq_len
        h = model.config.num_attention_heads
        # Attention: QK^T: 2·n·d², SV: 2·n·d², proj: n·d² = 5·n·d²
        # MLP (SwiGLU): 3 projections to ~8/3·d, 1 back = 4·n·(8/3)·d²
        attn_flops = 5 * n * d * d
        mlp_flops = 4 * n * (8*d/3) * d
        total = L * (attn_flops + mlp_flops) * batch
        total += n * model.config.vocab_size * d * batch
        return total
    return 0


def flops_str(flops):
    if flops > 1e12: return f"{flops/1e12:.2f}T"
    if flops > 1e9: return f"{flops/1e9:.2f}B"
    return f"{flops/1e6:.0f}M"


def create_equal_flops_models(target_flops_per_step=5e9, vocab=5000, hidden=256, seq_len=64):
    """Create models with approx equal FLOPs per step."""
    # TinyBrain: adjust max_think_steps
    tb_cfg = TinyBrainConfig(vocab_size=vocab, hidden_size=hidden, num_cells=3, memory_slots=16, num_think_heads=2, max_think_steps=4, min_think_steps=1, output_mlp_hidden=hidden*2)
    tb_m = TinyBrainModel(tb_cfg)
    tb_flops = estimate_flops_per_step(tb_m, seq_len)
    
    # Transformer: adjust num_layers to match
    tf_flops = 0
    tf_m = None
    for layers in [2, 3, 4, 6, 8, 12]:
        tf_cfg_try = NovaConfig(vocab_size=vocab, hidden_size=hidden, num_layers=layers, num_attention_heads=4, intermediate_size=hidden*3, max_seq_length=seq_len)
        tf_m_try = NovaModel(tf_cfg_try)
        tf_flops_try = estimate_flops_per_step(tf_m_try, seq_len)
        if abs(tf_flops_try - tb_flops) < abs(tf_flops - tb_flops) or tf_flops == 0:
            tf_flops = tf_flops_try
            tf_m = tf_m_try
    
    return tb_m.to(DEVICE), tf_m.to(DEVICE), tb_flops, tf_flops


def build_data(vocab=5000, n=5000, seq_len=64):
    """Build synthetic dataset."""
    torch.manual_seed(42)
    data = []
    for _ in range(n):
        if torch.rand(1).item() < 0.3:
            base = torch.randint(1, vocab//2, (1,)).item()
            data.append(torch.tensor([(base+i)%vocab for i in range(seq_len)], dtype=torch.long))
        else:
            data.append(torch.randint(1, vocab, (seq_len,)).long())
    split = int(n*0.9)
    class D(torch.utils.data.Dataset):
        def __init__(self, d): self.d = d
        def __len__(self): return len(self.d)
        def __getitem__(self,i): return self.d[i],self.d[i].clone()
    return D(data[:split]), D(data[split:])


def train(model, train_ds, val_ds, steps=500, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    tl = torch.utils.data.DataLoader(train_ds, 32, True)
    vl = torch.utils.data.DataLoader(val_ds, 32)
    history = []
    step = 0
    while step < steps:
        model.train()
        for x,y in tl:
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
                vl_=0.0
                with torch.no_grad():
                    for x2,y2 in vl:
                        x2,y2=x2.to(DEVICE),y2.to(DEVICE)
                        vl_+=model(x2,labels=y2)["loss"].item()
                vl_/=len(vl)
                history.append({"step":step,"val_loss":vl_})
                print(f"  step={step:4d}/{steps} | val_loss={vl_:.4f}")
                model.train()
    return history


def run():
    print("="*60)
    print("EQUAL-FLOPs BENCHMARK")
    print("="*60)
    
    # Create models
    tb, tf, tb_flops, tf_flops = create_equal_flops_models(target_flops_per_step=5e9)
    n_tb = sum(p.numel() for p in tb.parameters())
    n_tf = sum(p.numel() for p in tf.parameters())
    
    print(f"\nModels:")
    print(f"  TinyBrain:   {n_tb:>8,} params | {flops_str(tb_flops)} FLOPs/step | cells={tb.config.num_cells} | steps={tb.config.max_think_steps}")
    print(f"  Transformer: {n_tf:>8,} params | {flops_str(tf_flops)} FLOPs/step | layers={tf.config.num_layers}")
    print(f"  FLOPs ratio: {tb_flops/tf_flops:.2f}x (1.0 = equal)")
    
    # Data
    train_ds, val_ds = build_data()
    print(f"\nData: {len(train_ds)} train + {len(val_ds)} val")
    
    # Train both for same FLOPs budget
    # If tb_flops > tf_flops, train transformer more steps
    steps_tb = 500
    steps_tf = int(steps_tb * tb_flops / tf_flops)
    print(f"\nTraining: TB={steps_tb} steps | TF={steps_tf} steps (equal FLOPs)")
    
    t0 = time.time()
    tb_hist = train(tb, train_ds, val_ds, steps_tb)
    t1 = time.time()
    tf_hist = train(tf, train_ds, val_ds, steps_tf)
    t2 = time.time()
    
    # Results
    tb_final = tb_hist[-1]["val_loss"] if tb_hist else 0
    tf_final = tf_hist[-1]["val_loss"] if tf_hist else 0
    tb_time = t1 - t0
    tf_time = t2 - t1
    tb_tok_s = len(train_ds)*64*steps_tb/tb_time if tb_time>0 else 0
    tf_tok_s = len(train_ds)*64*steps_tf/tf_time if tf_time>0 else 0
    
    print(f"\n{'='*60}")
    print("EQUAL-FLOPs RESULTS")
    print(f"{'='*60}")
    print(f"\n{'Metric':<30} {'TinyBrain':<20} {'Transformer':<20}")
    print("-"*70)
    print(f"{'Parameters':<30} {n_tb:<20,} {n_tf:<20,}")
    print(f"{'FLOPs per step':<30} {flops_str(tb_flops):<20} {flops_str(tf_flops):<20}")
    print(f"{'Training steps':<30} {steps_tb:<20} {steps_tf:<20}")
    print(f"{'Total FLOPs':<30} {flops_str(tb_flops*steps_tb):<20} {flops_str(tf_flops*steps_tf):<20}")
    print(f"{'Final Val Loss':<30} {tb_final:<20.4f} {tf_final:<20.4f}")
    print(f"{'Training time':<30} {tb_time:<20.1f}s {tf_time:<20.1f}s")
    print(f"{'Tokens/sec':<30} {tb_tok_s:<20.0f} {tf_tok_s:<20.0f}")
    
    winner = "TinyBrain" if tb_final < tf_final else "Transformer"
    margin = abs(tb_final - tf_final)
    print(f"\n{'🏆 WINNER':<30} {'✅ '+winner if tb_final<tf_final else '✅ '+winner:<20}")
    print(f"{'Margin':<30} {margin:<20.4f}")
    
    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT / f"equal_flops_{ts}.json"
    with open(path,"w") as f:
        json.dump({
            "tinybrain": {"params":n_tb,"flops":tb_flops,"steps":steps_tb,"final_val":tb_final,"time":tb_time},
            "transformer": {"params":n_tf,"flops":tf_flops,"steps":steps_tf,"final_val":tf_final,"time":tf_time},
            "winner": winner if tb_final<tf_final else winner,
            "margin": margin,
        }, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
=======
>>>>>>> b7f35e8 (Diagnosis: TinyStories failure analysis + all benchmarks)
"""
Equal-FLOPs Benchmark — Fair Comparison

Goal: Compare TinyBrain vs Transformer at SAME compute budget.
Not same steps. Not same params. SAME FLOPS.

Method:
1. Measure FLOPs for 1 Transformer step
2. Count how many TinyBrain steps fit in same FLOPs
3. Compare loss at equal FLOPs budget
"""
import sys, os, math, time, json
from pathlib import Path
from datetime import datetime
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("tb", os.path.join(os.path.dirname(os.path.abspath(__file__)), "novacore", "models", "tiny_brain.py"))
tb = importlib.util.module_from_spec(spec)
sys.modules["tb"] = tb
spec.loader.exec_module(tb)
TinyBrainConfig, TinyBrainModel = tb.TinyBrainConfig, tb.TinyBrainModel

from novacore.core.simple_model import NovaModel
from novacore.core.config import NovaConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("novacore/experiments/equal_flops_results")
OUT.mkdir(parents=True, exist_ok=True)


def estimate_flops_per_step(model, seq_len=64, batch=1):
    """Estimate FLOPs per forward pass."""
    if isinstance(model, TinyBrainModel):
        d = model.config.hidden_size
        K = model.config.num_cells
        m = model.config.memory_slots
        T = model.config.max_think_steps
        n = seq_len
        # Each thinking step: ~5·n·d² (thinking) + 2·n·d² (memory project) + n·m·d (memory read)
        step_flops = 5 * n * d * d + 2 * n * d * d + n * m * d
        total = K * T * step_flops * batch
        # Add embed + output
        total += n * model.config.vocab_size * d * batch  # lm_head
        return total
    elif isinstance(model, NovaModel):
        d = model.config.hidden_size
        L = model.config.num_layers
        n = seq_len
        h = model.config.num_attention_heads
        # Attention: QK^T: 2·n·d², SV: 2·n·d², proj: n·d² = 5·n·d²
        # MLP (SwiGLU): 3 projections to ~8/3·d, 1 back = 4·n·(8/3)·d²
        attn_flops = 5 * n * d * d
        mlp_flops = 4 * n * (8*d/3) * d
        total = L * (attn_flops + mlp_flops) * batch
        total += n * model.config.vocab_size * d * batch
        return total
    return 0


def flops_str(flops):
    if flops > 1e12: return f"{flops/1e12:.2f}T"
    if flops > 1e9: return f"{flops/1e9:.2f}B"
    return f"{flops/1e6:.0f}M"


def create_equal_flops_models(target_flops_per_step=5e9, vocab=5000, hidden=256, seq_len=64):
    """Create models with approx equal FLOPs per step."""
    # TinyBrain: adjust max_think_steps
    tb_cfg = TinyBrainConfig(vocab_size=vocab, hidden_size=hidden, num_cells=3, memory_slots=16, num_think_heads=2, max_think_steps=4, min_think_steps=1, output_mlp_hidden=hidden*2)
    tb_m = TinyBrainModel(tb_cfg)
    tb_flops = estimate_flops_per_step(tb_m, seq_len)
    
    # Transformer: adjust num_layers to match
    tf_flops = 0
    tf_m = None
    for layers in [2, 3, 4, 6, 8, 12]:
        tf_cfg_try = NovaConfig(vocab_size=vocab, hidden_size=hidden, num_layers=layers, num_attention_heads=4, intermediate_size=hidden*3, max_seq_length=seq_len)
        tf_m_try = NovaModel(tf_cfg_try)
        tf_flops_try = estimate_flops_per_step(tf_m_try, seq_len)
        if abs(tf_flops_try - tb_flops) < abs(tf_flops - tb_flops) or tf_flops == 0:
            tf_flops = tf_flops_try
            tf_m = tf_m_try
    
    return tb_m.to(DEVICE), tf_m.to(DEVICE), tb_flops, tf_flops


def build_data(vocab=5000, n=5000, seq_len=64):
    """Build synthetic dataset."""
    torch.manual_seed(42)
    data = []
    for _ in range(n):
        if torch.rand(1).item() < 0.3:
            base = torch.randint(1, vocab//2, (1,)).item()
            data.append(torch.tensor([(base+i)%vocab for i in range(seq_len)], dtype=torch.long))
        else:
            data.append(torch.randint(1, vocab, (seq_len,)).long())
    split = int(n*0.9)
    class D(torch.utils.data.Dataset):
        def __init__(self, d): self.d = d
        def __len__(self): return len(self.d)
        def __getitem__(self,i): return self.d[i],self.d[i].clone()
    return D(data[:split]), D(data[split:])


def train(model, train_ds, val_ds, steps=500, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    tl = torch.utils.data.DataLoader(train_ds, 32, True)
    vl = torch.utils.data.DataLoader(val_ds, 32)
    history = []
    step = 0
    while step < steps:
        model.train()
        for x,y in tl:
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
                vl_=0.0
                with torch.no_grad():
                    for x2,y2 in vl:
                        x2,y2=x2.to(DEVICE),y2.to(DEVICE)
                        vl_+=model(x2,labels=y2)["loss"].item()
                vl_/=len(vl)
                history.append({"step":step,"val_loss":vl_})
                print(f"  step={step:4d}/{steps} | val_loss={vl_:.4f}")
                model.train()
    return history


def run():
    print("="*60)
    print("EQUAL-FLOPs BENCHMARK")
    print("="*60)
    
    # Create models
    tb, tf, tb_flops, tf_flops = create_equal_flops_models(target_flops_per_step=5e9)
    n_tb = sum(p.numel() for p in tb.parameters())
    n_tf = sum(p.numel() for p in tf.parameters())
    
    print(f"\nModels:")
    print(f"  TinyBrain:   {n_tb:>8,} params | {flops_str(tb_flops)} FLOPs/step | cells={tb.config.num_cells} | steps={tb.config.max_think_steps}")
    print(f"  Transformer: {n_tf:>8,} params | {flops_str(tf_flops)} FLOPs/step | layers={tf.config.num_layers}")
    print(f"  FLOPs ratio: {tb_flops/tf_flops:.2f}x (1.0 = equal)")
    
    # Data
    train_ds, val_ds = build_data()
    print(f"\nData: {len(train_ds)} train + {len(val_ds)} val")
    
    # Train both for same FLOPs budget
    # If tb_flops > tf_flops, train transformer more steps
    steps_tb = 500
    steps_tf = int(steps_tb * tb_flops / tf_flops)
    print(f"\nTraining: TB={steps_tb} steps | TF={steps_tf} steps (equal FLOPs)")
    
    t0 = time.time()
    tb_hist = train(tb, train_ds, val_ds, steps_tb)
    t1 = time.time()
    tf_hist = train(tf, train_ds, val_ds, steps_tf)
    t2 = time.time()
    
    # Results
    tb_final = tb_hist[-1]["val_loss"] if tb_hist else 0
    tf_final = tf_hist[-1]["val_loss"] if tf_hist else 0
    tb_time = t1 - t0
    tf_time = t2 - t1
    tb_tok_s = len(train_ds)*64*steps_tb/tb_time if tb_time>0 else 0
    tf_tok_s = len(train_ds)*64*steps_tf/tf_time if tf_time>0 else 0
    
    print(f"\n{'='*60}")
    print("EQUAL-FLOPs RESULTS")
    print(f"{'='*60}")
    print(f"\n{'Metric':<30} {'TinyBrain':<20} {'Transformer':<20}")
    print("-"*70)
    print(f"{'Parameters':<30} {n_tb:<20,} {n_tf:<20,}")
    print(f"{'FLOPs per step':<30} {flops_str(tb_flops):<20} {flops_str(tf_flops):<20}")
    print(f"{'Training steps':<30} {steps_tb:<20} {steps_tf:<20}")
    print(f"{'Total FLOPs':<30} {flops_str(tb_flops*steps_tb):<20} {flops_str(tf_flops*steps_tf):<20}")
    print(f"{'Final Val Loss':<30} {tb_final:<20.4f} {tf_final:<20.4f}")
    print(f"{'Training time':<30} {tb_time:<20.1f}s {tf_time:<20.1f}s")
    print(f"{'Tokens/sec':<30} {tb_tok_s:<20.0f} {tf_tok_s:<20.0f}")
    
    winner = "TinyBrain" if tb_final < tf_final else "Transformer"
    margin = abs(tb_final - tf_final)
    print(f"\n{'🏆 WINNER':<30} {'✅ '+winner if tb_final<tf_final else '✅ '+winner:<20}")
    print(f"{'Margin':<30} {margin:<20.4f}")
    
    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT / f"equal_flops_{ts}.json"
    with open(path,"w") as f:
        json.dump({
            "tinybrain": {"params":n_tb,"flops":tb_flops,"steps":steps_tb,"final_val":tb_final,"time":tb_time},
            "transformer": {"params":n_tf,"flops":tf_flops,"steps":steps_tf,"final_val":tf_final,"time":tf_time},
            "winner": winner if tb_final<tf_final else winner,
            "margin": margin,
        }, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
<<<<<<< HEAD
=======
>>>>>>> 22df365 (Fix equal-FLOPs benchmark attribute error)
>>>>>>> b7f35e8 (Diagnosis: TinyStories failure analysis + all benchmarks)
=======
"""
Equal-FLOPs Benchmark — Fair Comparison

Goal: Compare TinyBrain vs Transformer at SAME compute budget.
Not same steps. Not same params. SAME FLOPS.

Method:
1. Measure FLOPs for 1 Transformer step
2. Count how many TinyBrain steps fit in same FLOPs
3. Compare loss at equal FLOPs budget
"""
import sys, os, math, time, json
from pathlib import Path
from datetime import datetime
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("tb", os.path.join(os.path.dirname(os.path.abspath(__file__)), "novacore", "models", "tiny_brain.py"))
tb = importlib.util.module_from_spec(spec)
sys.modules["tb"] = tb
spec.loader.exec_module(tb)
TinyBrainConfig, TinyBrainModel = tb.TinyBrainConfig, tb.TinyBrainModel

from novacore.core.simple_model import NovaModel
from novacore.core.config import NovaConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("novacore/experiments/equal_flops_results")
OUT.mkdir(parents=True, exist_ok=True)


def estimate_flops_per_step(model, seq_len=64, batch=1):
    """Estimate FLOPs per forward pass."""
    if isinstance(model, TinyBrainModel):
        d = model.config.hidden_size
        K = model.config.num_cells
        m = model.config.memory_slots
        T = model.config.max_think_steps
        n = seq_len
        # Each thinking step: ~5·n·d² (thinking) + 2·n·d² (memory project) + n·m·d (memory read)
        step_flops = 5 * n * d * d + 2 * n * d * d + n * m * d
        total = K * T * step_flops * batch
        # Add embed + output
        total += n * model.config.vocab_size * d * batch  # lm_head
        return total
    elif isinstance(model, NovaModel):
        d = model.config.hidden_size
        L = model.config.num_layers
        n = seq_len
        h = model.config.num_attention_heads
        # Attention: QK^T: 2·n·d², SV: 2·n·d², proj: n·d² = 5·n·d²
        # MLP (SwiGLU): 3 projections to ~8/3·d, 1 back = 4·n·(8/3)·d²
        attn_flops = 5 * n * d * d
        mlp_flops = 4 * n * (8*d/3) * d
        total = L * (attn_flops + mlp_flops) * batch
        total += n * model.config.vocab_size * d * batch
        return total
    return 0


def flops_str(flops):
    if flops > 1e12: return f"{flops/1e12:.2f}T"
    if flops > 1e9: return f"{flops/1e9:.2f}B"
    return f"{flops/1e6:.0f}M"


def create_equal_flops_models(target_flops_per_step=5e9, vocab=5000, hidden=256, seq_len=64):
    """Create models with approx equal FLOPs per step."""
    # TinyBrain: adjust max_think_steps
    tb_cfg = TinyBrainConfig(vocab_size=vocab, hidden_size=hidden, num_cells=3, memory_slots=16, num_think_heads=2, max_think_steps=4, min_think_steps=1, output_mlp_hidden=hidden*2)
    tb_m = TinyBrainModel(tb_cfg)
    tb_flops = estimate_flops_per_step(tb_m, seq_len)
    
    # Transformer: adjust num_layers to match
    tf_flops = 0
    tf_m = None
    for layers in [2, 3, 4, 6, 8, 12]:
        tf_cfg_try = NovaConfig(vocab_size=vocab, hidden_size=hidden, num_layers=layers, num_attention_heads=4, intermediate_size=hidden*3, max_seq_length=seq_len)
        tf_m_try = NovaModel(tf_cfg_try)
        tf_flops_try = estimate_flops_per_step(tf_m_try, seq_len)
        if abs(tf_flops_try - tb_flops) < abs(tf_flops - tb_flops) or tf_flops == 0:
            tf_flops = tf_flops_try
            tf_m = tf_m_try
    
    return tb_m.to(DEVICE), tf_m.to(DEVICE), tb_flops, tf_flops


def build_data(vocab=5000, n=5000, seq_len=64):
    """Build synthetic dataset."""
    torch.manual_seed(42)
    data = []
    for _ in range(n):
        if torch.rand(1).item() < 0.3:
            base = torch.randint(1, vocab//2, (1,)).item()
            data.append(torch.tensor([(base+i)%vocab for i in range(seq_len)], dtype=torch.long))
        else:
            data.append(torch.randint(1, vocab, (seq_len,)).long())
    split = int(n*0.9)
    class D(torch.utils.data.Dataset):
        def __init__(self, d): self.d = d
        def __len__(self): return len(self.d)
        def __getitem__(self,i): return self.d[i],self.d[i].clone()
    return D(data[:split]), D(data[split:])


def train(model, train_ds, val_ds, steps=500, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    tl = torch.utils.data.DataLoader(train_ds, 32, True)
    vl = torch.utils.data.DataLoader(val_ds, 32)
    history = []
    step = 0
    while step < steps:
        model.train()
        for x,y in tl:
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
                vl_=0.0
                with torch.no_grad():
                    for x2,y2 in vl:
                        x2,y2=x2.to(DEVICE),y2.to(DEVICE)
                        vl_+=model(x2,labels=y2)["loss"].item()
                vl_/=len(vl)
                history.append({"step":step,"val_loss":vl_})
                print(f"  step={step:4d}/{steps} | val_loss={vl_:.4f}")
                model.train()
    return history


def run():
    print("="*60)
    print("EQUAL-FLOPs BENCHMARK")
    print("="*60)
    
    # Create models
    tb, tf, tb_flops, tf_flops = create_equal_flops_models(target_flops_per_step=5e9)
    n_tb = sum(p.numel() for p in tb.parameters())
    n_tf = sum(p.numel() for p in tf.parameters())
    
    print(f"\nModels:")
    print(f"  TinyBrain:   {n_tb:>8,} params | {flops_str(tb_flops)} FLOPs/step | cells={tb.config.num_cells} | steps={tb.config.max_think_steps}")
    print(f"  Transformer: {n_tf:>8,} params | {flops_str(tf_flops)} FLOPs/step | layers={tf.config.num_layers}")
    print(f"  FLOPs ratio: {tb_flops/tf_flops:.2f}x (1.0 = equal)")
    
    # Data
    train_ds, val_ds = build_data()
    print(f"\nData: {len(train_ds)} train + {len(val_ds)} val")
    
    # Train both for same FLOPs budget
    # If tb_flops > tf_flops, train transformer more steps
    steps_tb = 500
    steps_tf = int(steps_tb * tb_flops / tf_flops)
    print(f"\nTraining: TB={steps_tb} steps | TF={steps_tf} steps (equal FLOPs)")
    
    t0 = time.time()
    tb_hist = train(tb, train_ds, val_ds, steps_tb)
    t1 = time.time()
    tf_hist = train(tf, train_ds, val_ds, steps_tf)
    t2 = time.time()
    
    # Results
    tb_final = tb_hist[-1]["val_loss"] if tb_hist else 0
    tf_final = tf_hist[-1]["val_loss"] if tf_hist else 0
    tb_time = t1 - t0
    tf_time = t2 - t1
    tb_tok_s = len(train_ds)*64*steps_tb/tb_time if tb_time>0 else 0
    tf_tok_s = len(train_ds)*64*steps_tf/tf_time if tf_time>0 else 0
    
    print(f"\n{'='*60}")
    print("EQUAL-FLOPs RESULTS")
    print(f"{'='*60}")
    print(f"\n{'Metric':<30} {'TinyBrain':<20} {'Transformer':<20}")
    print("-"*70)
    print(f"{'Parameters':<30} {n_tb:<20,} {n_tf:<20,}")
    print(f"{'FLOPs per step':<30} {flops_str(tb_flops):<20} {flops_str(tf_flops):<20}")
    print(f"{'Training steps':<30} {steps_tb:<20} {steps_tf:<20}")
    print(f"{'Total FLOPs':<30} {flops_str(tb_flops*steps_tb):<20} {flops_str(tf_flops*steps_tf):<20}")
    print(f"{'Final Val Loss':<30} {tb_final:<20.4f} {tf_final:<20.4f}")
    print(f"{'Training time':<30} {tb_time:<20.1f}s {tf_time:<20.1f}s")
    print(f"{'Tokens/sec':<30} {tb_tok_s:<20.0f} {tf_tok_s:<20.0f}")
    
    winner = "TinyBrain" if tb_final < tf_final else "Transformer"
    margin = abs(tb_final - tf_final)
    print(f"\n{'🏆 WINNER':<30} {'✅ '+winner if tb_final<tf_final else '✅ '+winner:<20}")
    print(f"{'Margin':<30} {margin:<20.4f}")
    
    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT / f"equal_flops_{ts}.json"
    with open(path,"w") as f:
        json.dump({
            "tinybrain": {"params":n_tb,"flops":tb_flops,"steps":steps_tb,"final_val":tb_final,"time":tb_time},
            "transformer": {"params":n_tf,"flops":tf_flops,"steps":steps_tf,"final_val":tf_final,"time":tf_time},
            "winner": winner if tb_final<tf_final else winner,
            "margin": margin,
        }, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
>>>>>>> 04c6b57bea4e06026148c237c8d09be699e685d8
    run()