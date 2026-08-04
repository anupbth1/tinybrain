"""Control: same 20-example task with a same-size TRANSFORMER (run on GPU).

Isolates TinyBrain vs shared pipeline (BPE/windows/left-pad generation):
  TF 20/20  -> TinyBrain conditioning is the gap -> debug TB specifically
  TF ~1/20  -> shared pipeline is the problem -> fix pipeline first
Both batched (left-pad) and per-prompt (no-pad) generation are measured so the
left-pad hypothesis is tested in the same run.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

import scale_path as sp
sp.NO_SAVE = True
from scale_path import SeqDS, make_tf, generate_batch

t0 = time.time()
prompts, answers = sp.load_gsm8k(20, "train")
t_p, t_a = sp.load_gsm8k(20, "test")
tok = sp._build_gsm8k_bpe(prompts, answers, t_p, t_a, vocab_size=4096)
data = sp._make_gsm8k_lm_data(prompts, answers, tok, 128)
tl = torch.utils.data.DataLoader(SeqDS(data, 128, pad_id=tok.pad_token_id),
                                 batch_size=8, shuffle=True)
print(f"{len(data)} windows | vocab={len(tok)} | {time.time()-t0:.0f}s", flush=True)

m = make_tf(len(tok), hidden=512, layers=4, heads=8).to(sp.DEVICE)
n = sum(p.numel() for p in m.parameters())
print(f"TF params={n/1e6:.1f}M", flush=True)
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

def score(gens, tag):
    strict = loose = 0
    for i, (g, q) in enumerate(zip(gens, qids)):
        dec = tok.decode(g[len(q):], skip_special_tokens=True)
        gold = sp._gsm8k_ans(answers[i])
        strict += sp._num_match(sp._gsm8k_ans(dec), gold)
        loose += gold in dec
    print(f"  {tag}: strict={strict}/20 loose={loose}/20", flush=True)
    return strict, loose

m.eval()
qids = [tok.encode(f"Question: {q}\nAnswer: ") for q in prompts]
# batched left-pad generation (same path as eval_gsm8k)
g_batch = generate_batch(m, qids, max_new=160, temp=0.0, no_repeat_ngram=3,
                         eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
score(g_batch, "batched(left-pad)")

# per-prompt no-pad generation
g_single = []
for q in qids:
    ids = list(q)
    for _ in range(160):
        with torch.no_grad():
            logits = m(torch.tensor([ids], device=sp.DEVICE), last_only=True)["logits"][0, -1]
        logits[0] = logits[1] = float("-inf")  # pad/unk
        nxt = int(logits.argmax().item())
        ids.append(nxt)
        if nxt == tok.eos_token_id:
            break
    g_single.append(q + ids[len(q):])
score(g_single, "per-prompt(no-pad)")

for i, g in enumerate(g_single):
    dec = tok.decode(g[len(qids[i]):], skip_special_tokens=True)
    print(f"  Q{i+1:2d} gold={sp._gsm8k_ans(answers[i])!r} pred={dec[:64]!r}", flush=True)
print(f"TF_CONTROL_DONE | {time.time()-t0:.0f}s", flush=True)
