"""Scratch: exercise full mode_memory_ablation + mode_equal_flops with stubbed loaders."""
import argparse
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

args = argparse.Namespace(seeds="0,1", steps=8, batch=8, samples=150, dataset="tinystories",
                          log_every=4, memory_sharp=None, lr=1e-3, warmup=0.3, early_stop=0)

print("=========== MEMORY ABLATION ===========")
r = sp.mode_memory_ablation(args)
assert r["summary"]["verdict"] in ("MEMORY_USED READ_NO_SLOT_ID WRITE_NO_SLOT_ID",
                                   "MEMORY_USED READ_NO_SLOT_ID WRITE_USES_SLOT_ID",
                                   "MEMORY_USED READ_USES_SLOT_ID WRITE_NO_SLOT_ID",
                                   "MEMORY_USED READ_USES_SLOT_ID WRITE_USES_SLOT_ID",
                                   "MEMORY_STILL_WEAK"), r["summary"]["verdict"]
assert set(r["per_seed"].keys()) == {"0", "1"}
assert "ro_delta_shuf_vals" in r["ablation"]
print("ablation verdict:", r["summary"]["verdict"])

print("\n=========== EQUAL FLOPS ===========")
r2 = sp.mode_equal_flops(args)
assert "stat_sig" in r2["summary"] and "p_value_paired_t" in r2["summary"]
assert "tf_tokens" in r2["seeds"]["0"] and "v2_tokens" in r2["seeds"]["0"]
assert "last_seed_internals" in r2 and "mem_eff_scale_mean" in r2["last_seed_internals"]
print("equal_flops summary:", {k: v for k, v in r2["summary"].items() if k != "cohens_d"})

print("\nMODE_SMOKE_OK")
