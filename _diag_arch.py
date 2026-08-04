"""Architecture instrumentation: 20-example overfit + routing/hidden/logit report.

Run on GPU (~5 min). A/B knobs:
  python _diag_arch.py                      # baseline (attn 0.5x, 16 slots)
  python _diag_arch.py --attn_ratio=1.0     # full-width attention A/B
  python _diag_arch.py --mem_slots=64       # slot count A/B

Report (saved + printed): per-cell gamma/gate/logit_scale, memory slot mass
(top-5 + entropy), attention entropy at the answer position, confidence mean,
hidden norms (mean + answer position), first-token top-5 + logit entropy,
and strict/loose memorization.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import torch.nn.functional as F

import scale_path as sp
from scale_path import SeqDS, make_tb, generate_batch

attn_ratio = None
mem_slots = None
final_norm = None
for a in sys.argv[1:]:
    if a.startswith("--attn_ratio="):
        attn_ratio = float(a.split("=")[1])
    if a.startswith("--mem_slots="):
        mem_slots = int(a.split("=")[1])
    if a.startswith("--final_norm="):
        final_norm = int(a.split("=")[1])

t0 = time.time()
prompts, answers = sp.load_gsm8k(20, "train")
t_p, t_a = sp.load_gsm8k(20, "test")
tok = sp._build_gsm8k_bpe(prompts, answers, t_p, t_a, vocab_size=4096)
data = sp._make_gsm8k_lm_data(prompts, answers, tok, 128)
tl = torch.utils.data.DataLoader(SeqDS(data, 128, pad_id=tok.pad_token_id),
                                 batch_size=8, shuffle=True)
print(f"{len(data)} windows | vocab={len(tok)}", flush=True)

m = make_tb(len(tok), "hybrid_v2", model_size="small",
            attn_ratio=attn_ratio, mem_slots=mem_slots, final_norm=final_norm)
print(f"TB params={sum(p.numel() for p in m.parameters())/1e6:.1f}M "
      f"attn_ratio={m.config.attn_dim_ratio} mem_slots={m.config.memory_slots} "
      f"final_norm={m.config.final_norm} "
      f"attn_dim={max(m.config.attn_heads, int(m.config.hidden_size*m.config.attn_dim_ratio))}", flush=True)
for c in m.cells:
    c.min_s = c.max_s = m.config.max_think_steps
    c.conf.thresh = 1.5
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
m.train()
last = None
for step in range(300):
    for x, y in tl:
        x, y = x.to(sp.DEVICE), y.to(sp.DEVICE)
        opt.zero_grad()
        loss = m(x, labels=y)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        last = loss.item()
    if (step + 1) % 50 == 0:
        print(f"  step {step+1:3d} loss={last:.4f} ({time.time()-t0:.0f}s)", flush=True)
print(f"SFT done: loss={last:.4f}", flush=True)

# ---------- instrumentation forward ----------
m.eval()
qids = [tok.encode(f"Question: {q}\nAnswer: ") for q in prompts[:8]]
L = max(len(q) for q in qids)
cur = torch.zeros(len(qids), L, dtype=torch.long, device=sp.DEVICE)
for i, q in enumerate(qids):
    cur[i, L - len(q):] = torch.tensor(q, dtype=torch.long, device=sp.DEVICE)
pm = cur != tok.pad_token_id

norms = {}
def make_hook(name):
    def f(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        norms[name + "_mean"] = round(x.norm(dim=-1).mean().item(), 4)
        norms[name + "_lastpos"] = round(x.norm(dim=-1)[0, -1].item(), 4)
    return f
hooks = [c.register_forward_hook(make_hook(f"cell{i}")) for i, c in enumerate(m.cells)]
m.out_mlp.register_forward_hook(make_hook("out_mlp"))
if m.final_norm is not None:
    m.final_norm.register_forward_hook(make_hook("final_norm"))
with torch.no_grad():
    m(cur, pad_mask=pm)
for h in hooks:
    h.remove()

rep = {"meta": {"attn_ratio": m.config.attn_dim_ratio, "mem_slots": m.config.memory_slots,
                "hidden": m.config.hidden_size, "final_norm": m.config.final_norm,
                "params": sum(p.numel() for p in m.parameters()),
                "train_loss": last}, "norms": norms, "cells": {}}
for i, c in enumerate(m.cells):
    gamma = torch.tanh(c.think.gamma).item()
    gate = torch.tanh(c.mem.out_gate).item()
    lscale = F.softplus(c.mem.logit_scale).item()
    a_mem = c.mem._last_attn  # (B,S,slots)
    mass = a_mem.mean(dim=(0, 1))
    top5 = mass.topk(min(5, mass.numel()))
    ent_slots = -(mass * mass.clamp_min(1e-9).log()).sum().item()
    a_attn = c.cell_attn._last_attn  # (B,H,S,S)
    a_pos = a_attn[0, :, -1, :]  # answer-start position
    ent_attn = -(a_pos * a_pos.clamp_min(1e-9).log()).sum(dim=-1).mean().item()
    conf = c.conf._last_c.mean().item()
    rep["cells"][f"cell{i}"] = {
        "gamma": round(gamma, 4), "out_gate": round(gate, 4), "logit_scale": round(lscale, 3),
        "slot_mass_top5": [round(v, 4) for v in top5.values.tolist()],
        "slot_entropy": round(ent_slots, 4),
        "attn_entropy_at_answer": round(ent_attn, 4),
        "conf_mean": round(conf, 4),
    }
    print(f"  cell{i}: gamma={gamma:.3f} gate={gate:.3f} lscale={lscale:.1f} "
          f"slots_top5={[round(v,3) for v in top5.values.tolist()]} slot_ent={ent_slots:.2f} "
          f"attn_ent={ent_attn:.2f} conf={conf:.3f}", flush=True)
print("  norms:", norms, flush=True)

# ---------- first-token logit diagnostics ----------
def first_token(qid):
    c1 = torch.zeros(1, len(qid), dtype=torch.long, device=sp.DEVICE)
    c1[0] = torch.tensor(qid, dtype=torch.long, device=sp.DEVICE)
    with torch.no_grad():
        logits = m(c1, last_only=True, pad_mask=c1 != tok.pad_token_id)["logits"][0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    top5 = probs.topk(5)
    ent = -(probs * probs.clamp_min(1e-9).log()).sum().item()
    return [tok.decode([int(i)], skip_special_tokens=True) for i in top5.indices], \
           [round(v, 4) for v in top5.values.tolist()], round(ent, 4)

rep["first_token"] = {}
for j, q in enumerate(qids[:3]):
    w, v, e = first_token(q)
    rep["first_token"][f"q{j+1}"] = {"top5": w, "probs": v, "logit_entropy": e}
    print(f"  q{j+1} first-token top5={w} probs={v} ent={e}", flush=True)

# ---------- generation ----------
qall = [tok.encode(f"Question: {q}\nAnswer: ") for q in prompts]
gen = generate_batch(m, qall, max_new=160, temp=0.0, no_repeat_ngram=3,
                     eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
strict = loose = 0
rows = []
for i, (g, q) in enumerate(zip(gen, qall)):
    dec = tok.decode(g[len(q):], skip_special_tokens=True)
    gold = sp._gsm8k_ans(answers[i])
    hit = sp._num_match(sp._gsm8k_ans(dec), gold)
    strict += hit
    loose += gold in dec
    rows.append({"gold": gold, "pred": dec[:160], "strict": hit, "contains": loose})
    print(f"  Q{i+1:2d} gold={gold!r} strict={hit} pred={dec[:56]!r}", flush=True)
rep["memorization"] = {"strict": strict, "loose": loose, "total": 20}
sp._save(rep, f"diag_arch_r{str(m.config.attn_dim_ratio).replace('.', '')}_s{m.config.memory_slots}")
print(f"DIAG_DONE strict={strict}/20 loose={loose}/20 | {time.time()-t0:.0f}s", flush=True)
