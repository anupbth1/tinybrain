# RunPod / Colab — Scale Path (1B ≈ 600B+ feel)

## Goal (honest framing)

600B models win by **depth of mixing + knowledge capacity**.
TinyBrain's bet: **params stay ~1B**, extra intelligence comes from
**iterative think-steps + memory + light token mixing**.

That only works if:
1. Small-scale quality matches Transformer (gap → ~0)
2. **More think-steps ⇒ better loss** at fixed params (compute scaling)
3. Then grow to 1B params with a large think budget

---

## What just landed (Hybrid results)

| Model | ValLoss | Δ vs TF |
|-------|---------|---------|
| Transformer | 4.0663 | 0 |
| TB-plain | 4.4287 | +0.36 |
| TB-hybrid v1 | 4.2878 | +0.22 |

Hybrid helps. Next: **v2** (attn before each cell + step-conditioned think).

---

## Colab / RunPod commands (use `!` in notebook)

```python
!git clone https://github.com/anupbth1/tinybrain.git
%cd tinybrain
!pip install -q datasets
!git pull
!python scale_path.py --verify
```

### Experiment order (run in order, paste RESULTS each time)

```python
# 1) Close the gap? (~15-40 min GPU)
!python scale_path.py --mode race --steps 2000

# 2) Are iterations actually refining? (~10-20 min)
!python scale_path.py --mode diagnose --steps 1000

# 3) Critical for 1B=600B: more think ⇒ better? (~20-40 min)
!python scale_path.py --mode think_scale --steps 800
```

---

## Decision tree

| Race result | Action |
|-------------|--------|
| v2 gap ≤ 0.10 vs TF | Go to think_scale |
| v2 better than v1 but gap > 0.15 | Longer train (5k) or wider attn |
| v2 ≤ v1 | Revert placement; try post+wider only |

| think_scale result | Action |
|--------------------|--------|
| T8 < T1 (COMPUTE_SCALES) | Scale params toward 50M→1B |
| T8 ≥ T1 | Fix iteration diversity before any scaling |

---

## How 1B gets 600B+ *feel*

Not magic compression of weights. Mechanism:

```
Transformer 600B:  quality ≈ f(params, layers, data)
TinyBrain 1B:      quality ≈ f(params, think_steps, memory, data)
```

If `∂quality/∂think_steps > 0`, you can spend FLOPs at inference
like a deeper model without storing 600B weights.
That is the only credible path to "1B feels like 600B+".

Until think_scale says COMPUTE_SCALES, do **not** jump to 1B training.
