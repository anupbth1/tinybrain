"""
Deep Diagnosis: Find why ThinkingStep and Memory contribute nothing.

Checks:
1. Gamma gradient flow — does gradient reach gamma?
2. Memory read vs write — are writes happening?
3. Layer activation stats — mean, std, norm, saturation per layer
4. Residual bypass — is residual dominating?
"""
import sys, os, json, math, time, torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util

spec = importlib.util.spec_from_file_location("tb", "novacore/models/tiny_brain.py")
tb = importlib.util.module_from_spec(spec)
sys.modules["tb"] = tb; spec.loader.exec_module(tb)
TinyBrainConfig, TinyBrainModel = tb.TinyBrainConfig, tb.TinyBrainModel
from novacore.core.simple_model import NovaModel
from novacore.core.config import NovaConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RES = Path("novacore/experiments/deep_diagnosis")
RES.mkdir(parents=True, exist_ok=True)

def analyze(model, x, name="model"):
    """Analyze model internals: gamma gradients, memory, activations."""
    model.train()
    x = x.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Forward + backward
    opt.zero_grad()
    out = model(x, labels=x)
    loss = out["loss"]
    loss.backward()
    
    results = {}
    
    # 1. Gamma gradient flow
    gammas = []
    gamma_grads = []
    for n, p in model.named_parameters():
        if "gamma" in n:
            gammas.append(p.item())
            gamma_grads.append(p.grad.norm().item() if p.grad is not None else 0.0)
    results["gamma"] = {
        "mean_value": sum(gammas)/len(gammas) if gammas else 0,
        "mean_grad": sum(gamma_grads)/len(gamma_grads) if gamma_grads else 0,
        "max_grad": max(gamma_grads) if gamma_grads else 0,
        "values": gammas,
        "grads": gamma_grads,
    }
    
    # 2. Memory analysis
    mem_norms_before = []
    mem_norms_after = []
    for n, p in model.named_parameters():
        if "M0" in n:  # Memory slots
            mem_norms_before.append(p.norm(dim=-1).mean().item())
            # Check grad on memory
            if p.grad is not None:
                mem_norms_after.append(p.grad.norm(dim=-1).mean().item())
    
    results["memory"] = {
        "mean_slot_norm": sum(mem_norms_before)/len(mem_norms_before) if mem_norms_before else 0,
        "num_slots": len(mem_norms_before),
        "grad_norm": sum(mem_norms_after)/len(mem_norms_after) if mem_norms_after else 0,
    }
    
    # 3. Output gate analysis
    out_gates = []
    out_gate_grads = []
    for n, p in model.named_parameters():
        if "out_gate" in n:
            out_gates.append(p.item())
            out_gate_grads.append(p.grad.norm().item() if p.grad is not None else 0.0)
    results["out_gate"] = {
        "mean_value": sum(out_gates)/len(out_gates) if out_gates else 0,
        "mean_grad": sum(out_gate_grads)/len(out_gate_grads) if out_gate_grads else 0,
    }
    
    # 4. Residual bypass analysis (forward pass)
    with torch.no_grad():
        model.eval()
        orig = model.embed(x)
        
        # Track norm through cells
        cell_norms = []
        h = orig.clone()
        for i, cell in enumerate(model.cells):
            h_in = h.norm().item()
            h, mem, aux = cell(h)
            h_out = h.norm().item()
            cell_norms.append({"cell": i, "in_norm": h_in, "out_norm": h_out, "delta": h_out - h_in})
        
        results["residual"] = {
            "embed_norm": orig.norm().item(),
            "final_norm": h.norm().item(),
            "cells": cell_norms,
            "total_delta": h.norm().item() - orig.norm().item(),
        }
    
    return results

def train_and_diagnose(steps=100):
    """Train TinyBrain tiny and diagnose internals."""
    print("="*60)
    print("DEEP DIAGNOSIS")
    print("="*60)
    
    # Tiny model
    cfg = TinyBrainConfig(vocab_size=200, hidden_size=64, num_cells=2, memory_slots=4, max_think_steps=4, output_mlp_hidden=128)
    model = TinyBrainModel(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Data
    torch.manual_seed(42)
    data = torch.randint(2, 200, (100, 64))
    all_results = []
    
    for step in range(steps):
        x = data[step % len(data)].unsqueeze(0)
        opt.zero_grad()
        out = model(x, labels=x)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if step % 20 == 0:
            diag = analyze(model, x)
            diag["step"] = step
            diag["loss"] = loss.item()
            all_results.append(diag)
            g = diag["gamma"]
            m = diag["memory"]
            r = diag["residual"]
            print(f"  step={step:3d} | loss={loss.item():.4f} | γ_val={g['mean_value']:.6f} | γ_grad={g['mean_grad']:.6f} | mem_norm={m['mean_slot_norm']:.4f} | mem_grad={m['grad_norm']:.6f} | out_gate={diag['out_gate']['mean_value']:.6f} | res_delta={r['total_delta']:.4f}")
    
    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RES / f"deep_diagnosis_{ts}.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {path}")
    
    # Summary
    last = all_results[-1] if all_results else {}
    print("\n=== DIAGNOSIS SUMMARY ===")
    print(f"Loss: {last.get('loss', 0):.4f}")
    print(f"Gamma gradient: {last.get('gamma', {}).get('mean_grad', 0):.6f} {'✅ FLOWING' if last.get('gamma', {}).get('mean_grad', 0) > 0.001 else '❌ NEAR ZERO'}")
    print(f"Memory gradient: {last.get('memory', {}).get('grad_norm', 0):.6f} {'✅ LEARNING' if last.get('memory', {}).get('grad_norm', 0) > 0.001 else '❌ NOT LEARNING'}")
    print(f"Residual delta: {last.get('residual', {}).get('total_delta', 0):.4f} {'✅ CHANGING' if abs(last.get('residual', {}).get('total_delta', 0)) > 0.1 else '❌ STUCK'}")
    print(f"Out gate: {last.get('out_gate', {}).get('mean_value', 0):.6f}")
    return all_results

if __name__ == "__main__":
    train_and_diagnose(100)