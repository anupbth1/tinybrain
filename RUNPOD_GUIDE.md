# RunPod / Colab — Path to low-cost 1B ≈ 600B+ feel

## Honest status (after your latest runs)

| Signal | Result |
|--------|--------|
| claim_holds (3/3 seeds, best val) | **True** |
| Memory selective? | **Yes** — top1≈0.22, ent_ratio≈0.80–0.84 (was uniform 0.0625) |
| iter_cos | ~0.97 (was 0.99) — better, still room |
| COMPUTE_SCALES T1→T8 | **stronger** (−1.89) |
| V2 best ≈1.46 vs TF ≈3.93 | **suspicious** on 5k data — likely extra FLOPs + memorization |

Do **not** treat 1.46 as production quality yet. Next gates are fair.

---

## Run next (in order)

```python
%cd tinybrain
!git pull
!python scale_path.py --verify
```

### 1) Memory ablation — is memory causal?
```python
!python scale_path.py --mode memory_ablation --steps 1500
```
Want: `MEMORY_USED` with Δ(zero−full) > 0.05

### 2) Equal-FLOPs on more data — real efficiency claim
```python
!python scale_path.py --mode equal_flops --steps 2000 --samples 20000
```
Auto-matches total FLOPs (V2 gets fewer steps because costlier/step).
Want: `v2_wins_equal_flops=True` (even if margin shrinks)

Paste both RESULTS blocks back.

---

## Then (only if both pass)

1. Associative-recall / needle task (memory’s real job)
2. Scale 5M → 50M → 1B
3. Inference: raise T on hard tokens only → cheap “600B feel”

## Goal reminder

600B companies pay for **params × FLOPs**.
Your wedge: **small params + spend FLOPs as think-steps** when needed.
That only ships if equal-FLOPs still wins and memory ablation shows MEMORY_USED.
