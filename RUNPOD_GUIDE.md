# RunPod / Colab Deployment Guide

## Critical test right now: Hybrid vs Transformer

Diagnosis showed TinyBrain loses to Transformer on TinyStories because it has
**no cross-token interaction**. Hybrid TinyBrain adds one lightweight causal
attention block after thinking cells (~10% extra FLOPs).

### On RunPod / Colab

```bash
# Install
pip install torch datasets

# Smoke test (~10s)
python hybrid_compare.py --verify

# Main experiment (~10-20 min on GPU)
python hybrid_compare.py --steps 500

# Stronger signal (~20-40 min)
python hybrid_compare.py --steps 1000
```

### What to paste back

The script prints a `RESULTS` table like:

```
Model                   Params  ValLoss   Δ vs TF    Time
Transformer            .......   4.xxxx   +0.0000   ...
TB-plain               .......   4.xxxx   +0.xxxx   ...
TB-hybrid              .......   4.xxxx   +0.xxxx   ...
Hybrid vs Transformer: ...
Hybrid vs Plain:       ...
Hybrid gates: γ=...  out_gate=...
```

**Copy that whole block + the JSON path** and send it.

### How to read it

| Result | Meaning | Next move |
|--------|---------|-----------|
| Hybrid ≤ Transformer | Cross-token fix worked | Scale up / equal-FLOPs |
| Hybrid << Plain, but still > TF | Attn helps, not enough | Attn per cell or wider attn |
| Hybrid ≈ Plain | Attn not learning | Check W_o grads / lr |
| γ / out_gate stay ~0.1 | Gates stuck | Raise init or lr on gates |

JSON is saved to: `novacore/experiments/hybrid_results/hybrid_*.json`

---

## Older modes (still valid)

```bash
python runpod_ready.py --mode verify
python runpod_ready.py --mode proof
python runpod_ready.py --mode ablation
```

## Files to download after a run

```
novacore/experiments/hybrid_results/hybrid_*.json
checkpoints/   # if you used runpod_ready proof mode
```
