"""Decisive 20-example overfit test at model_size=small (run on GPU, ~5 min).

If the model memorizes the 20 train Q->A pairs (loss -> ~0.1 and generation
reproduces the #### answers), the pipeline + architecture are PROVEN correct
and GSM8K test acc=0 is purely the toy-scale generalization ceiling.
If it can't (0-5/20), there is a real conditioning bug worth deep-diving.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

import scale_path as sp
sp.NO_SAVE = True
from scale_path import SeqDS, make_tb, generate_batch

t0 = time.time()
prompts, answers = sp.load_gsm8k(20, "train")
t_p, t_a = sp.load_gsm8k(20, "test")
tok = sp._build_gsm8k_bpe(prompts, answers, t_p, t_a, vocab_size=4096)
data = sp._make_gsm8k_lm_data(prompts, answers, tok, 128)
tl = torch.utils.data.DataLoader(SeqDS(data, 128, pad_id=tok.pad_token_id),
                                 batch_size=8, shuffle=True)
print(f"{len(data)} windows | vocab={len(tok)} | {time.time()-t0:.0f}s", flush=True)

m = make_tb(len(tok), "hybrid_v2", model_size="small")  # 15M
# pin full depth for BOTH train and eval (deterministic, matches real run)
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

# generate on the SAME 20 train questions, long enough to reach '#### <num>'
m.eval()
qids = [tok.encode(f"Question: {q}\nAnswer: ") for q in prompts]
gen = generate_batch(m, qids, max_new=160, temp=0.0, no_repeat_ngram=3,
                     eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
strict = loose = 0
for i, (g, q) in enumerate(zip(gen, qids)):
    dec = tok.decode(g[len(q):], skip_special_tokens=True)
    gold = sp._gsm8k_ans(answers[i])
    hit = sp._num_match(sp._gsm8k_ans(dec), gold)
    contains = gold in dec
    strict += hit
    loose += contains
    print(f"  Q{i+1:2d} gold={gold!r} strict={hit} contains={contains} pred={dec[:64]!r}", flush=True)
print(f"MEMORIZATION: strict={strict}/20 loose={loose}/20 | {time.time()-t0:.0f}s", flush=True)
