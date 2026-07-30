"""
Run all TinyBrain experiments in sequence.
Outputs results to novacore/experiments/results/
"""
import sys, os, json, math, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct import of tiny_brain (self-contained, no package deps)
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
ThinkingStep = tb.ThinkingStep
LearnedMemory = tb.LearnedMemory
ConfidenceGate = tb.ConfidenceGate
AdaptiveThinkingCell = tb.AdaptiveThinkingCell
SelfCorrection = tb.SelfCorrection

RESULTS = Path(os.path.dirname(os.path.abspath(__file__))) / "novacore" / "experiments" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
DEVICE = "cpu"
import torch
import numpy as np

def load_results(filename):
    """Load existing results if any."""
    p = RESULTS / filename
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}

def save_results(filename, data):
    """Save results."""
    with open(RESULTS / filename, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {filename}")

LOG = []

def log(name, status, details=""):
    LOG.append({"name": name, "status": status, "details": details, "time": datetime.now().isoformat()})
    print(f"  [{status:>5}] {name}")

def log_md():
    """Write experiment_log.md"""
    lines = ["# TinyBrain Experiment Log", f"\nLast updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
    
    for entry in LOG:
        lines.append(f"## {entry['name']}")
        lines.append(f"- **Status**: {entry['status']}")
        if entry['details']:
            lines.append(f"- **Details**: {entry['details']}")
        lines.append("")
    
    with open(RESULTS / "experiment_log.md", "w") as f:
        f.write("\n".join(lines))

print("=" * 60)
print("TinyBrain — Complete Test Suite")
print("=" * 60)

# ─── TEST 1: Basic Model Creation ───
print("\n[1] Basic Model Creation")
cfg = TinyBrainConfig(hidden_size=64, num_cells=2, memory_slots=4)
m = TinyBrainModel(cfg)
n = sum(p.numel() for p in m.parameters())
print(f"  Params: {n:,}")
x = torch.randint(0, 100, (2, 16))
o = m(x, labels=x)
print(f"  Loss: {o['loss'].item():.4f}")
log("Model creation", "PASS", f"{n:,} params, loss={o['loss'].item():.4f}")

# ─── TEST 2: ThinkingStep Convergence ───
print("\n[2] ThinkingStep Convergence")
step = ThinkingStep(cfg)
xt = torch.randn(4, 16, 64)
deltas = []
for t in range(100):
    xn = step(xt)
    d = (xn - xt).norm(dim=-1).mean().item()
    deltas.append(d); xt = xn
ratio = deltas[-1] / max(deltas[0], 1e-10)
print(f"  Initial: {deltas[0]:.6f}, Final: {deltas[-1]:.6f}, Ratio: {ratio:.6f}")
log("ThinkingStep convergence", "PASS" if ratio < 0.01 else "FAIL", f"ratio={ratio:.6f}")

# ─── TEST 3: LearnedMemory ───
print("\n[3] LearnedMemory")
mem = LearnedMemory(cfg)
x = torch.randn(2, 8, 64)
r, state = mem(x)
print(f"  Output: {r.shape}, State: {state[0].shape}")
log("LearnedMemory", "PASS" if r.shape == (2,8,64) else "FAIL")

# ─── TEST 4: ConfidenceGate ───
print("\n[4] ConfidenceGate")
gate = ConfidenceGate(cfg)
x0 = torch.randn(2, 8, 64)
x1 = x0 + 0.5 * torch.randn(2, 8, 64)
c, h = gate(x1, x0, 5)
print(f"  Confidence: {c.mean().item():.4f}, Halt: {h.mean().item():.4f}")
log("ConfidenceGate", "PASS" if c.shape == (2,8,1) else "FAIL")

# ─── TEST 5: AdaptiveThinkingCell ───
print("\n[5] AdaptiveThinkingCell")
cell = AdaptiveThinkingCell(cfg)
x = torch.randn(2, 8, 64)
out, mem_state, aux = cell(x)
print(f"  Output: {out.shape}, Steps: {aux.get('steps','?')}")
log("AdaptiveThinkingCell", "PASS" if out.shape == (2,8,64) else "FAIL")

# ─── TEST 6: Full Model Forward + Loss ───
print("\n[6] Full Model Forward")
m = TinyBrainModel(cfg)
x = torch.randint(0, 100, (4, 32))
o = m(x, labels=x)
assert "loss" in o, "No loss"
assert o["loss"].item() > 0, "Loss should be positive"
log("Full Model", "PASS", f"loss={o['loss'].item():.4f}")

# ─── TEST 7: Memory Persistence ───
print("\n[7] Memory Persistence")
m.eval()
x1 = torch.randint(0, 100, (1, 16))
o1 = m(x1)
mem1 = o1["memory_states"]
o2 = m(x1, memory_states=mem1)  # Should work with cached memory
assert "loss" in o2 or "logits" in o2, "Memory persistence failed"
log("Memory persistence", "PASS")

# ─── TEST 8: Thinking Step Dynamics (key graph) ───
print("\n[8] Thinking Steps vs Perplexity (key graph)")
cfg2 = TinyBrainConfig(hidden_size=64, num_cells=2, memory_slots=4, max_think_steps=32, min_think_steps=1)
cell2 = AdaptiveThinkingCell(cfg2)
cell2.eval()
x = torch.randn(4, 16, 64)
for t in [2, 4, 8, 16, 32]:
    # Override max_steps
    cell2.max_s = t
    out, _, _ = cell2(x)
    perp = out.pow(2).mean().item()  # proxy for perplexity
    print(f"  {t:2d} steps: loss={perp:.4f}")
log("Thinking steps sweep", "INFO", "2,4,8,16,32 steps tested")

# ─── TEST 9: Memory Slot Sweep ───
print("\n[9] Memory Slots vs Performance")
for m_slots in [4, 8, 16]:
    cfg_m = TinyBrainConfig(hidden_size=64, num_cells=2, memory_slots=m_slots)
    mem_m = LearnedMemory(cfg_m)
    x = torch.randn(2, 16, 64)
    r, state = mem_m(x)
    print(f"  {m_slots:2d} slots: output={r.shape}")
log("Memory slot sweep", "INFO", "4,8,16 slots tested")

# ─── TEST 10: Confidence Threshold Sweep ───
print("\n[10] Confidence Threshold Sweep")
cell3 = AdaptiveThinkingCell(cfg2)
cell3.eval()
x = torch.randn(4, 16, 64)
full_out, _, _ = cell3(x)
for thresh in [0.5, 0.7, 0.85, 0.95]:
    cell3.conf.thresh = thresh
    out, _, aux = cell3(x)
    err = (out - full_out).norm().item()
    print(f"  threshold={thresh:.2f}: error={err:.4f}")
log("Confidence threshold sweep", "INFO", "0.5,0.7,0.85,0.95 tested")

# ─── SAVE ALL RESULTS ───
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_results(f"test_results_{timestamp}.json", LOG)
log_md()

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
passed = sum(1 for l in LOG if l['status'] == 'PASS')
failed = sum(1 for l in LOG if l['status'] == 'FAIL')
print(f"  PASSED: {passed}")
print(f"  FAILED: {failed}")
print(f"  TOTAL:  {len(LOG)}")
print(f"  Results: {RESULTS}")
print("=" * 60)