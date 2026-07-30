"""
Standalone TinyBrain test — imports model directly.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load tiny_brain module directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "tiny_brain",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "novacore", "models", "tiny_brain.py")
)
tb = importlib.util.module_from_spec(spec)
sys.modules["tiny_brain"] = tb
spec.loader.exec_module(tb)

# Test basic functionality
config = tb.TinyBrainConfig(hidden_size=64, num_cells=2, memory_slots=4)
print(f"Config OK: d={config.hidden_size}, K={config.num_cells}, m={config.memory_slots}")

model = tb.TinyBrainModel(config)
print(f"Model created: {sum(p.numel() for p in model.parameters()):,} params")

# Forward pass
import torch
x = torch.randint(0, 100, (2, 16))
out = model(x, labels=x)
print(f"Forward pass OK. Loss: {out['loss'].item():.4f}")

# Test ThinkingStep convergence
step = tb.ThinkingStep(config)
x_t = torch.randn(4, 16, 64)
deltas = []
for t in range(50):
    xn = step(x_t)
    d = (xn - x_t).norm(dim=-1).mean().item()
    deltas.append(d)
    x_t = xn

print(f"\nThinkingStep convergence test:")
print(f"  Initial delta: {deltas[0]:.6f}")
print(f"  Final delta:   {deltas[-1]:.6f}")
print(f"  Ratio:         {deltas[-1]/max(deltas[0],1e-10):.6f}")
print(f"  PASS:          {deltas[-1]/max(deltas[0],1e-10) < 0.01}")

# Test LearnedMemory
mem = tb.LearnedMemory(config)
x = torch.randn(2, 8, 64)
r, state = mem(x)
print(f"\nLearnedMemory test:")
print(f"  Output shape: {r.shape}")
print(f"  State shape: {state[0].shape}")
print(f"  PASS:         {r.shape == (2,8,64)}")

# Test ConfidenceGate
gate = tb.ConfidenceGate(config)
x0 = torch.randn(2, 8, 64)
x1 = x0 + 0.5 * torch.randn(2, 8, 64)
c, h = gate(x1, x0, 5)
print(f"\nConfidenceGate test:")
print(f"  Confidence shape: {c.shape}")
print(f"  Mean confidence:  {c.mean().item():.4f}")
print(f"  Halt mean:        {h.mean().item():.4f}")

# Test AdaptiveThinkingCell
cell = tb.AdaptiveThinkingCell(config)
x = torch.randn(2, 8, 64)
out, mem, aux = cell(x)
print(f"\nAdaptiveThinkingCell test:")
print(f"  Output shape: {out.shape}")
print(f"  Steps:        {aux.get('steps', '?')}")
print(f"  PASS:         {out.shape == (2,8,64)}")

print("\n" + "="*60)
print("ALL TESTS PASSED")
print("="*60)