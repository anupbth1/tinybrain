"""Smoke: thought_paths (multi-agent), label smoothing, amp flag, code_eval pieces."""
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
tl = torch.utils.data.DataLoader(SeqDS(data[:split], 64), batch_size=8, shuffle=True)
vl = torch.utils.data.DataLoader(SeqDS(data[split:], 64), batch_size=8)

# 1) mixture-of-thoughts: K=2 forward/backward + FLOPs increase
m1 = make_tb(VOCAB, "hybrid_v2", think_steps=4, paths=1)
m2 = make_tb(VOCAB, "hybrid_v2", think_steps=4, paths=2)
f1, f2 = measure_fwd_flops(m1), measure_fwd_flops(m2)
print(f"paths=1 flops={f1:,} paths=2 flops={f2:,} ratio={f2/f1:.2f}")
assert f2 > f1
x = torch.randint(0, VOCAB, (2, 64))
out = m2(x, labels=x)
assert math.isfinite(out["loss"].item())
out["loss"].backward()
print("paths=2 forward+backward OK, params:", sum(p.numel() for p in m2.parameters()))

# 2) training with paths=2 + label smoothing + amp flag (CPU-safe)
m = make_tb(VOCAB, "hybrid_v2", think_steps=4, paths=2)
m.label_smoothing = 0.05
r = train_one(m, tl, vl, steps=6, name="paths2", log_every=3, lr=1e-3, warmup_fraction=0.3,
              amp=True, seq_len=64)
print("paths=2 train best=%.4f" % r["best_val_loss"])
assert r["best_val_loss"] > 0

# 3) generate + _check_code with a trivial toy
tok = {"<pad>": 0, "<unk>": 1, "<eos>": 2, "b97": 3, "b98": 4, "b10": 5, "b32": 6}
mm = make_tb(VOCAB, "hybrid_v2", think_steps=2)
pid = [3, 4]
gen = sp.generate(mm, tok, pid, max_new=10, context=16)
print("generate len:", len(gen))
assert len(gen) >= len(pid)
ok = sp._check_code("def f():\n    return 4\n", "assert f() == 4\n")
bad = sp._check_code("def f():\n    return 3\n", "assert f() == 4\n")
print("check_code ok=%s bad=%s" % (ok, bad))
assert ok and not bad

# 4) equal_flops with paths + seq_len via stubbed loaders
def fake_loaders(args, seed=0):
    return tl, vl, VOCAB

sp.get_loaders = fake_loaders
args = argparse.Namespace(seeds="0", steps=8, batch=8, samples=150, dataset="tinystories",
                          data_mix=None, seq_len=64, thought_paths=2, label_smooth=0.0, amp=False,
                          log_every=4, memory_sharp=None, lr=1e-3, warmup=0.3, early_stop=0,
                          think_steps=4, tf_layers=3, think_rank=None, ema=0.0, max_new=64)
r2 = sp.mode_equal_flops(args)
assert r2["seeds"]["0"]["flops_v2"] > r2["seeds"]["0"]["flops_tf"]
print("equal_flops(paths=2) Δ=%.4f ratio=%.2f" % (r2["summary"]["delta_mean"], r2["seeds"]["0"]["flops_v2"] / r2["seeds"]["0"]["flops_tf"]))

# 5) GSM8K pieces: answer match + generate(temp) + rollout_logprobs + full GRPO loop (stubbed data)
assert sp._num_match("123", "123") and not sp._num_match("124", "123")
assert sp._num_match("1,234", "1234") and sp._num_match("5.0", "5")
assert sp._gsm8k_ans("x #### 42") == "42"

import argparse
import torch as T

def fake_gsm8k(max_samples=20000, split="train"):
    qs = [f"what is {i} plus {i}" for i in range(24)]
    ans = [f"#### {2 * i}" for i in range(24)]
    return qs, ans

sp.load_gsm8k = fake_gsm8k
rl_args = argparse.Namespace(
    seeds="0", steps=2, batch=8, samples=24, dataset="gsm8k", data_mix=None, seq_len=64,
    thought_paths=1, label_smooth=0.0, amp=False, log_every=10, memory_sharp=None, lr=1e-3,
    warmup=0.3, early_stop=0, think_steps=2, tf_layers=3, think_rank=None, ema=0.0,
    max_new=16, rl_steps=2, rollouts=2, rl_batch=4, rl_max_new=12, rl_lr=1e-4, rl_temp=0.9,
    rl_kl=0.01, rl_pretrain=2, reason_samples=1, compile=False,
)
res = sp.mode_grpo(rl_args)
assert "gsm8k_acc_after_rl" in res
print("GRPO loop OK, acc_after_rl =", res["gsm8k_acc_after_rl"])

# 6) generate_batch: all sequences advance in lockstep, same length
mm2 = make_tb(VOCAB, "hybrid_v2", think_steps=2)
gb = sp.generate_batch(mm2, [[3, 4], [5, 6, 7]], max_new=6, temp=0.0)
print("generate_batch lens:", [len(g) for g in gb])
assert len(gb) == 2 and all(len(g) >= 6 for g in gb)

print("ALL_SMOKE_OK")
