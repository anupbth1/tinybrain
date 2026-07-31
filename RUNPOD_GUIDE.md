# RunPod / Colab — Path to 1B ≈ 600B+ feel (low cost)

## Goal

Others run 600B+ for quality (huge GPU bill).
Your bet: **~1B params + more think-steps at inference** ≈ same feel, far lower train/run cost.

Only valid if: quality scales with **T (think steps)**, not just params.

---

## What the last results proved

| Signal | Status |
|--------|--------|
| Hybrid v2 beats TF (even after TF overfit caveat) | strong |
| T=1→8 lowers loss (COMPUTE_SCALES) | **core thesis alive** |
| Memory was uniform (top1=1/16) | **bug — fixed now** |
| iter_cos ≈ 0.99 | **wasted steps — diversity loss added** |
| Report **best** val, not final (TF rose 3.94→4.71) | fixed in scripts |

Fair re-read of your race (best, not final):
- TF best ≈ **3.94** @800
- V2 best ≈ **3.37** @1800  
Still a real win, just not the fake −1.28 from TF collapse.

---

## Fixes just pushed

1. **Selective memory write** — slots no longer all get the same mean vector
2. **Sharp read** — learnable logit scale
3. **Diversity aux loss** — penalize iter cosine > 0.95
4. **best_val tracking** + **verify_claim** multi-seed

---

## Run next (Colab: use `!`)

```python
%cd tinybrain
!git pull
!python scale_path.py --verify
```

**1) Did memory + diversity fix work?** (~10–20 min)

```python
!python scale_path.py --mode diagnose --steps 1000
```

Want: `top1 > 0.15`, `mem_entropy_ratio < 0.85`, `iter_cos < 0.98`

**2) Does the win reproduce?** (~45–90 min, 3 seeds)

```python
!python scale_path.py --mode verify_claim --steps 2000 --seeds 0,1,2
```

Want: `claim_holds=True` on **best** val

**3) Re-check compute scaling after fixes**

```python
!python scale_path.py --mode think_scale --steps 800
```

Want: still `COMPUTE_SCALES`

Paste each RESULTS block back.

---

## After claim_holds

1. Equal-FLOPs curve (same compute budget, not same steps)
2. Scale 5M → 50M → 1B params
3. Inference: raise T on hard tokens only (dynamic compute = cheap “600B feel”)

Do **not** jump to 1B training until verify_claim passes.
