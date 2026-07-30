<<<<<<< HEAD
"""
Diagnose TinyBrain failure on TinyStories.
Run this on RunPod.
"""
import sys, os, json, math, time
from pathlib import Path
from datetime import datetime

import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util

spec = importlib.util.spec_from_file_location("tb", os.path.join(os.path.dirname(os.path.abspath(__file__)), "novacore", "models", "tiny_brain.py"))
tb = importlib.util.module_from_spec(spec)
sys.modules["tb"] = tb; spec.loader.exec_module(tb)
TinyBrainConfig, TinyBrainModel = tb.TinyBrainConfig, tb.TinyBrainModel

from novacore.core.simple_model import NovaModel
from novacore.core.config import NovaConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RES = Path("novacore/experiments/diagnosis_results")
RES.mkdir(parents=True, exist_ok=True)


def load_tinystories(max_samples=5000):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train")
    texts = ds["text"][:max_samples]
    words = set()
    for t in texts:
        for w in t.lower().split()[:50]: words.add(w)
    vl = sorted(words)
    w2i = {w:i+2 for i,w in enumerate(vl)}
    w2i["<pad>"]=0; w2i["<unk>"]=1
    def tok(t):
        return [w2i.get(w,1) for w in t.lower().split()[:64]]
    data = [torch.tensor(tok(t), dtype=torch.long) for t in texts if len(t.split())>5]
    return data, len(w2i)


def quick_train(model, tl, vl, steps=300, name="model"):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    t0 = time.time()
    step = 0
    while step < steps:
        model.train()
        for x,y in tl:
            if step>=steps: break
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad()
            out=model(x,labels=y)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            step+=1
    t1=time.time()
    model.eval()
    vl_=0.0
    with torch.no_grad():
        for x2,y2 in vl:
            x2,y2=x2.to(DEVICE),y2.to(DEVICE)
            vl_+=model(x2,labels=y2)["loss"].item()
    vl_/=len(vl)
    print(f"  {name:25s}: val_loss={vl_:.4f} | time={t1-t0:.1f}s")
    return vl_, t1-t0


def run():
    print("="*60)
    print("TinyStories DIAGNOSIS")
    print("="*60)
    
    # Data
    print("\nLoading data...")
    data, vocab = load_tinystories(5000)
    split=int(len(data)*0.9)
    from torch.utils.data import Dataset, DataLoader
    class D(Dataset):
        def __init__(self,d):self.d=d
        def __len__(s):return len(s.d)
        def __getitem__(s,i):
            x=s.d[i]
            if x.numel()<64: x=torch.cat([x,torch.zeros(64-x.numel(),dtype=torch.long)])
            return x[:64],x[:64].clone()
    tl=DataLoader(D(data[:split]),32,True)
    vl=DataLoader(D(data[split:]),32)
    print(f"  {len(data)} seqs, vocab={vocab}")
    
    # Baseline: Transformer
    print("\n--- BASELINE ---")
    tf_cfg = NovaConfig(vocab_size=vocab, hidden_size=256, num_layers=3, num_attention_heads=4, intermediate_size=768, max_seq_length=64)
    tf_m = NovaModel(tf_cfg).to(DEVICE)
    quick_train(tf_m, tl, vl, 300, "Transformer")
    
    # Full TinyBrain
    print("\n--- FULL TINYBRAIN ---")
    tb_cfg = TinyBrainConfig(vocab_size=vocab, hidden_size=256, num_cells=3, memory_slots=16, num_think_heads=2, max_think_steps=4, output_mlp_hidden=512)
    tb_m = TinyBrainModel(tb_cfg).to(DEVICE)
    quick_train(tb_m, tl, vl, 300, "Full TinyBrain")
    
    # H1: No ThinkingStep (gamma=0)
    print("\n--- H1: NO THINKINGSTEP ---")
    h1_m = TinyBrainModel(TinyBrainConfig(vocab_size=vocab, hidden_size=256, num_cells=3, memory_slots=16, num_think_heads=2, max_think_steps=4, output_mlp_hidden=512)).to(DEVICE)
    for n,p in h1_m.named_parameters():
        if "gamma" in n: p.data.zero_(); p.requires_grad=False
    quick_train(h1_m, tl, vl, 300, "H1: No ThinkingStep")
    
    # H2: No Memory (out_gate=0)
    print("\n--- H2: NO MEMORY ---")
    h2_m = TinyBrainModel(TinyBrainConfig(vocab_size=vocab, hidden_size=256, num_cells=3, memory_slots=16, num_think_heads=2, max_think_steps=4, output_mlp_hidden=512)).to(DEVICE)
    for n,p in h2_m.named_parameters():
        if "out_gate" in n: p.data.zero_(); p.requires_grad=False
    quick_train(h2_m, tl, vl, 300, "H2: No Memory")
    
    # H3: Fixed steps (2,4,8,16)
    print("\n--- H3: FIXED STEPS ---")
    for s in [2,4,8,16]:
        cfg_s = TinyBrainConfig(vocab_size=vocab, hidden_size=256, num_cells=3, memory_slots=16, num_think_heads=2, max_think_steps=s, min_think_steps=s, output_mlp_hidden=512)
        m_s = TinyBrainModel(cfg_s).to(DEVICE)
        quick_train(m_s, tl, vl, 300, f"H3: always {s} steps")
    
    # H4: Memory read-only (write gate frozen at 0)
    print("\n--- H4: MEMORY READ-ONLY ---")
    h4_m = TinyBrainModel(TinyBrainConfig(vocab_size=vocab, hidden_size=256, num_cells=3, memory_slots=16, num_think_heads=2, max_think_steps=4, output_mlp_hidden=512)).to(DEVICE)
    for n,p in h4_m.named_parameters():
        if "W_w" in n: p.data.zero_(); p.requires_grad=False  # freeze write
    quick_train(h4_m, tl, vl, 300, "H4: Memory read-only")
    
    print("\n"+"="*60)
    print("DONE. Check novacore/experiments/diagnosis_results/")
    print("="*60)

if __name__=="__main__":
=======
"""
Diagnose TinyBrain failure on TinyStories.
Run this on RunPod.
"""
import sys, os, json, math, time
from pathlib import Path
from datetime import datetime

import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util

spec = importlib.util.spec_from_file_location("tb", os.path.join(os.path.dirname(os.path.abspath(__file__)), "novacore", "models", "tiny_brain.py"))
tb = importlib.util.module_from_spec(spec)
sys.modules["tb"] = tb; spec.loader.exec_module(tb)
TinyBrainConfig, TinyBrainModel = tb.TinyBrainConfig, tb.TinyBrainModel

from novacore.core.simple_model import NovaModel
from novacore.core.config import NovaConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RES = Path("novacore/experiments/diagnosis_results")
RES.mkdir(parents=True, exist_ok=True)


def load_tinystories(max_samples=5000):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train")
    texts = ds["text"][:max_samples]
    words = set()
    for t in texts:
        for w in t.lower().split()[:50]: words.add(w)
    vl = sorted(words)
    w2i = {w:i+2 for i,w in enumerate(vl)}
    w2i["<pad>"]=0; w2i["<unk>"]=1
    def tok(t):
        return [w2i.get(w,1) for w in t.lower().split()[:64]]
    data = [torch.tensor(tok(t), dtype=torch.long) for t in texts if len(t.split())>5]
    return data, len(w2i)


def quick_train(model, tl, vl, steps=300, name="model"):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    t0 = time.time()
    step = 0
    while step < steps:
        model.train()
        for x,y in tl:
            if step>=steps: break
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad()
            out=model(x,labels=y)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            step+=1
    t1=time.time()
    model.eval()
    vl_=0.0
    with torch.no_grad():
        for x2,y2 in vl:
            x2,y2=x2.to(DEVICE),y2.to(DEVICE)
            vl_+=model(x2,labels=y2)["loss"].item()
    vl_/=len(vl)
    print(f"  {name:25s}: val_loss={vl_:.4f} | time={t1-t0:.1f}s")
    return vl_, t1-t0


def run():
    print("="*60)
    print("TinyStories DIAGNOSIS")
    print("="*60)
    
    # Data
    print("\nLoading data...")
    data, vocab = load_tinystories(5000)
    split=int(len(data)*0.9)
    from torch.utils.data import Dataset, DataLoader
    class D(Dataset):
        def __init__(self,d):self.d=d
        def __len__(s):return len(s.d)
        def __getitem__(s,i):
            x=s.d[i]
            if x.numel()<64: x=torch.cat([x,torch.zeros(64-x.numel(),dtype=torch.long)])
            return x[:64],x[:64].clone()
    tl=DataLoader(D(data[:split]),32,True)
    vl=DataLoader(D(data[split:]),32)
    print(f"  {len(data)} seqs, vocab={vocab}")
    
    # Baseline: Transformer
    print("\n--- BASELINE ---")
    tf_cfg = NovaConfig(vocab_size=vocab, hidden_size=256, num_layers=3, num_attention_heads=4, intermediate_size=768, max_seq_length=64)
    tf_m = NovaModel(tf_cfg).to(DEVICE)
    quick_train(tf_m, tl, vl, 300, "Transformer")
    
    # Full TinyBrain
    print("\n--- FULL TINYBRAIN ---")
    tb_cfg = TinyBrainConfig(vocab_size=vocab, hidden_size=256, num_cells=3, memory_slots=16, num_think_heads=2, max_think_steps=4, output_mlp_hidden=512)
    tb_m = TinyBrainModel(tb_cfg).to(DEVICE)
    quick_train(tb_m, tl, vl, 300, "Full TinyBrain")
    
    # H1: No ThinkingStep (gamma=0)
    print("\n--- H1: NO THINKINGSTEP ---")
    h1_m = TinyBrainModel(TinyBrainConfig(vocab_size=vocab, hidden_size=256, num_cells=3, memory_slots=16, num_think_heads=2, max_think_steps=4, output_mlp_hidden=512)).to(DEVICE)
    for n,p in h1_m.named_parameters():
        if "gamma" in n: p.data.zero_(); p.requires_grad=False
    quick_train(h1_m, tl, vl, 300, "H1: No ThinkingStep")
    
    # H2: No Memory (out_gate=0)
    print("\n--- H2: NO MEMORY ---")
    h2_m = TinyBrainModel(TinyBrainConfig(vocab_size=vocab, hidden_size=256, num_cells=3, memory_slots=16, num_think_heads=2, max_think_steps=4, output_mlp_hidden=512)).to(DEVICE)
    for n,p in h2_m.named_parameters():
        if "out_gate" in n: p.data.zero_(); p.requires_grad=False
    quick_train(h2_m, tl, vl, 300, "H2: No Memory")
    
    # H3: Fixed steps (2,4,8,16)
    print("\n--- H3: FIXED STEPS ---")
    for s in [2,4,8,16]:
        cfg_s = TinyBrainConfig(vocab_size=vocab, hidden_size=256, num_cells=3, memory_slots=16, num_think_heads=2, max_think_steps=s, min_think_steps=s, output_mlp_hidden=512)
        m_s = TinyBrainModel(cfg_s).to(DEVICE)
        quick_train(m_s, tl, vl, 300, f"H3: always {s} steps")
    
    # H4: Memory read-only (write gate frozen at 0)
    print("\n--- H4: MEMORY READ-ONLY ---")
    h4_m = TinyBrainModel(TinyBrainConfig(vocab_size=vocab, hidden_size=256, num_cells=3, memory_slots=16, num_think_heads=2, max_think_steps=4, output_mlp_hidden=512)).to(DEVICE)
    for n,p in h4_m.named_parameters():
        if "W_w" in n: p.data.zero_(); p.requires_grad=False  # freeze write
    quick_train(h4_m, tl, vl, 300, "H4: Memory read-only")
    
    print("\n"+"="*60)
    print("DONE. Check novacore/experiments/diagnosis_results/")
    print("="*60)

if __name__=="__main__":
>>>>>>> 04c6b57bea4e06026148c237c8d09be699e685d8
    run()