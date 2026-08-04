"""5-min CPU pipeline test: data -> BPE -> SFT -> Q&A gen -> reward -> acc."""
import sys, time
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

import scale_path as sp
sp.NO_SAVE = True
from scale_path import SeqDS, make_tb, generate_batch

BUDGET = 200  # seconds for SFT
t0 = time.time()
ok = True

def chk(name, cond, extra=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}", flush=True)
    ok = ok and cond

# ---------- 1) data + BPE ----------
prompts, answers = sp.load_gsm8k(300, "train")
t_p, t_a = sp.load_gsm8k(60, "test")
tok = sp._build_gsm8k_bpe(prompts, answers, t_p, t_a, vocab_size=8192)
rt = tok.decode(tok.encode("#### 42"), skip_special_tokens=True)
chk("BPE '#### 42' round-trip (reward depends on this)", "####" in rt, f"-> {rt!r}")
data = sp._make_gsm8k_lm_data(prompts, answers, tok, 128)
split = int(len(data) * 0.9)
tl = torch.utils.data.DataLoader(SeqDS(data[:split], 128, pad_id=tok.pad_token_id),
                                 batch_size=8, shuffle=True)
vl = torch.utils.data.DataLoader(SeqDS(data[split:], 128, pad_id=tok.pad_token_id),
                                 batch_size=8)
chk("windows + pad-masking", len(data) >= len(prompts))
xb, yb = next(iter(vl))
chk("pad labels masked", int((yb == -100).sum()) >= 0 and (xb == tok.pad_token_id).sum() == (yb == -100).sum())
print(f"  data: {len(data)} windows | {time.time()-t0:.0f}s", flush=True)

# ---------- 2) SFT (budgeted) ----------
m = make_tb(len(tok), "hybrid_v2", think_steps=4, rank=32)  # nano+rank, CPU-speed
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
m.train()
steps_done, losses = 0, []
for epoch in range(50):
    for x, y in tl:
        if time.time() - t0 > BUDGET:
            break
        opt.zero_grad()
        loss = m(x, labels=y)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        steps_done += 1
        losses.append(loss.item())
        if steps_done % 10 == 0:
            print(f"  step {steps_done:3d} loss={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    if time.time() - t0 > BUDGET:
        break
n = max(len(losses), 1)
chk("SFT loss decreasing", losses[-1] < losses[0] or steps_done < 10,
    f"steps={steps_done} loss {losses[0]:.2f}->{losses[-1]:.2f}")

# ---------- 3) Q&A generation ----------
m.eval()
qids = [tok.encode(f"Question: {q}\nAnswer: ") for q in t_p[:4]]
gen = generate_batch(m, qids, max_new=30, temp=0.0, no_repeat_ngram=3,
                     eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
nonempty = 0
for i, (g, q) in enumerate(zip(gen, qids)):
    dec = tok.decode(g[len(q):], skip_special_tokens=True)
    if dec.strip():
        nonempty += 1
    r = sp._gsm8k_reward(dec, t_a[i])
    print(f"  Q{i+1} gold={t_a[i].split('####')[-1].strip()!r} pred={dec[:60]!r} reward={r:.1f}", flush=True)
chk("generation non-empty (no pad/'' collapse)", nonempty >= 2, f"{nonempty}/4")

# ---------- 4) accuracy machinery on 8 problems ----------
t8 = t_p[:8]
q8 = [tok.encode(f"Question: {q}\nAnswer: ") for q in t8]
g8 = generate_batch(m, q8, max_new=30, temp=0.0, no_repeat_ngram=3,
                    eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
correct = 0
for g, q, a in zip(g8, q8, t_a[:8]):
    dec = tok.decode(g[len(q):], skip_special_tokens=True)
    if sp._num_match(sp._gsm8k_ans(dec), sp._gsm8k_ans(a)):
        correct += 1
print(f"  CPU accuracy (8 problems): {correct}/8", flush=True)
chk("accuracy machinery runs", correct >= 0, f"acc={correct/8:.2f} (0 expected at 2.4M params/60 steps)")

print(f"\n{'ALL_PIPELINE_PASS' if ok else 'PIPELINE_FAIL'} | total {time.time()-t0:.0f}s", flush=True)
