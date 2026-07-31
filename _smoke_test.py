"""Scratch: exercise new paths — measured FLOPs, think_steps/tf_layers, n=1 stats."""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

import scale_path as sp
from scale_path import SeqDS

torch.manual_seed(0)
VOCAB = 300
data = [torch.randint(1, VOCAB, (64,)).long() for _ in range(150)]
split = 130
tl = torch.utils.data.DataLoader(SeqDS(data[:split]), batch_size=8, shuffle=True)
vl = torch.utils.data.DataLoader(SeqDS(data[split:]), batch_size=8)


def fake_loaders(args, seed=0):
    return tl, vl, VOCAB


sp.get_loaders = fake_loaders

# 1) measure_fwd_flops works and is finite
tf = sp.make_tf(VOCAB)
v2 = sp.make_tb(VOCAB, "hybrid_v2", think_steps=8)
mf_tf = sp.measure_fwd_flops(tf)
mf_v2 = sp.measure_fwd_flops(v2)
print(f"measured flops tf={mf_tf:,} v2(T8)={mf_v2:,} ratio={mf_v2/mf_tf:.3f}")
assert mf_tf and mf_v2 and math.isfinite(mf_tf) and math.isfinite(mf_v2)

# 2) paired_stats with n=1 → finite p (1.0), no crash
ps1 = sp.paired_stats([3.5], [3.2])
print("n=1 stats:", ps1)
assert ps1["p_value_paired_t"] == 1.0 and ps1["sign_test_p"] == 1.0

# 3) equal_flops with think_steps/tf_layers via stubbed loaders
args = argparse.Namespace(seeds="0", steps=8, batch=8, samples=150, dataset="tinystories",
                          log_every=4, memory_sharp=None, lr=1e-3, warmup=0.3, early_stop=0,
                          think_steps=8, tf_layers=3)
r = sp.mode_equal_flops(args)
seed0 = r["seeds"]["0"]
assert seed0["think_steps"] == 8 and seed0["tf_layers"] == 3
assert seed0["flops_tf"] > 0 and seed0["flops_v2"] > 0
assert r["summary"]["p_value_paired_t"] == 1.0  # n=1 → p=1.0
print("equal_flops(T8) summary ok:", r["summary"]["delta_mean"], "| wins:", r["summary"]["v2_wins"])
print("FLOPs", seed0["flops_tf"], seed0["flops_v2"])

print("FIX_SMOKE_OK")
