"""Measure true per-step FLOPs with torch's FlopCounterMode."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from torch.utils.flop_counter import FlopCounterMode

import scale_path as sp

torch.manual_seed(0)
VOCAB = 18195
SEQ = 64

tf = sp.make_tf(VOCAB)
v2 = sp.make_tb(VOCAB, "hybrid_v2")

x = torch.randint(0, VOCAB, (1, SEQ))

for name, m in [("tf", tf), ("v2", v2)]:
    m.eval()
    with torch.no_grad():
        with FlopCounterMode(display=False) as fm:
            m(x, labels=x)
    flops = fm.get_total_flops()
    print(f"{name}: measured={flops:,}  estimate={sp.estimate_fwd_flops(m):,}  ratio_meas/est={flops/sp.estimate_fwd_flops(m):.2f}")

mt, mv = None, None
with torch.no_grad():
    with FlopCounterMode(display=False) as fm:
        tf(x, labels=x)
    mt = fm.get_total_flops()
with torch.no_grad():
    with FlopCounterMode(display=False) as fm:
        v2(x, labels=x)
    mv = fm.get_total_flops()
print(f"measured ratio v2/tf = {mv/mt:.3f}")
print(f"estimate ratio v2/tf = {sp.estimate_fwd_flops(v2)/sp.estimate_fwd_flops(tf):.3f}")
