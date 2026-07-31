"""Smoke: think_rank (low-rank thinking) + EMA + full mode with new args."""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

import scale_path as sp
from scale_path import SeqDS, make_tb, train_one, measure_fwd_flops

torch.manual_seed(0)
VOCAB = 300
data = [torch.randint(1, VOCAB, (64,)).long() for _ in range(150)]
split = 130
tl = torch.utils.data.DataLoader(SeqDS(data[:split]), batch_size=8, shuffle=True)
vl = torch.utils.data.DataLoader(SeqDS(data[split:]), batch_size=8)

# 1) think_rank: FLOPs must drop vs full-rank, forward/backward must work
m_full = make_tb(VOCAB, "hybrid_v2", think_steps=8)
m_lr = make_tb(VOCAB, "hybrid_v2", think_steps=8, rank=32)
f_full = measure_fwd_flops(m_full)
f_lr = measure_fwd_flops(m_lr)
print(f"T8 flops full-rank={f_full:,} low-rank(32)={f_lr:,} reduction={f_lr/f_full:.2f}x")
assert f_lr < f_full, "low-rank must be cheaper"
x = torch.randint(0, VOCAB, (2, 64))
out = m_lr(x, labels=x)
assert math.isfinite(out["loss"].item())
out["loss"].backward()
print("low-rank forward+backward OK, params:", sum(p.numel() for p in m_lr.parameters()))

# 2) EMA training runs
r = train_one(make_tb(VOCAB, "hybrid_v2", rank=32), tl, vl, steps=8, name="ema_t",
              log_every=4, lr=1e-3, warmup_fraction=0.3, ema=0.99)
print("ema result best=%.4f ema_flag=%s" % (r["best_val_loss"], r["ema"]))
assert r["ema"] == 0.99

# 3) full equal_flops with think_rank + ema + stubbed loaders
def fake_loaders(args, seed=0):
    return tl, vl, VOCAB

sp.get_loaders = fake_loaders
args = argparse.Namespace(seeds="0", steps=8, batch=8, samples=150, dataset="tinystories",
                          log_every=4, memory_sharp=None, lr=1e-3, warmup=0.3, early_stop=0,
                          think_steps=8, tf_layers=3, think_rank=32, ema=0.0)
r2 = sp.mode_equal_flops(args)
assert r2["seeds"]["0"]["flops_v2"] < f_full  # low-rank flops used
print("equal_flops(rank32) Δ=%.4f flops_v2=%d" % (r2["summary"]["delta_mean"], r2["seeds"]["0"]["flops_v2"]))

print("ALL_SMOKE_OK")
