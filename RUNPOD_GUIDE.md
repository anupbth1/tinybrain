# RunPod / Colab Deployment Guide

## Quick Start (2 minutes to training)

### Step 1: Copy to RunPod

```bash
# On your local machine, zip the project:
zip -r tinybrain.zip C:/project3/novacore C:/project3/runpod_ready.py C:/project3/run_all.py

# Upload to RunPod via their file manager, or:
# If using GitHub:
git init
git add .
git commit -m "TinyBrain project"
git remote add origin https://github.com/your/tinybrain.git
git push -u origin main
```

### Step 2: On RunPod / Colab

```bash
# Clone or upload
git clone https://github.com/your/tinybrain.git
cd tinybrain

# Install dependencies
pip install torch numpy

# Verify everything works (2 seconds)
python runpod_ready.py --mode verify
# Expected output:
#   ✅ TinyBrain 5M    | 3,239,790 params
#   ✅ Transformer 5M  | 3,641,600 params
#   ✅ All models verified
```

### Step 3: Run 5M Proof-of-Concept

```bash
# Full 5M test with checkpoints every 250 steps
python runpod_ready.py --mode proof
# Expected time: ~5-10 minutes on GPU (vs 1+ hour on CPU)
# 
# Output:
#   [tinybrain] step   250/2000 | val_loss=6.2 | γ=0.02
#   [tinybrain] step   500/2000 | val_loss=5.8 | γ=0.04
#   [tinybrain] step  1000/2000 | val_loss=5.4 | γ=0.06
#   [tinybrain] step  2000/2000 | val_loss=5.0 | γ=0.08
#
# Checkpoints saved to: checkpoints/tinybrain_step_*.pt
```

### Step 4: Ablation Study

```bash
# Which components actually help?
python runpod_ready.py --mode ablation
# ~15 minutes on GPU
```

### Step 5: Continue Training

```bash
# If training was interrupted, resume from latest checkpoint:
python runpod_ready.py --mode continue

# Or explicitly:
python runpod_ready.py --mode continue --checkpoint checkpoints/tinybrain_step_1000.pt
```

## Expected Results Template

After training, you'll see a table like this:

```
Step  | TB Loss | TF Loss | TB γ    | TB Gate | TB Mem
------|---------|---------|---------|---------|--------
   0  |   8.83  |   8.61  |  0.0000 |  0.0000 | 100.00%
 250  |   ?     |   ?     |  ?      |  ?      |  ?
 500  |   ?     |   ?     |  ?      |  ?      |  ?
1000  |   ?     |   ?     |  ?      |  ?      |  ?
2000  |   ?     |   ?     |  ?      |  ?      |  ?
```

**Fill in the ? marks and send them to me.** I'll analyze whether TinyBrain is learning.

## Interpretation Guide

| Pattern | Meaning |
|---------|---------|
| TB loss < TF loss | ✅ TinyBrain is better |
| TB loss ≈ TF loss | 🤝 Tie — check other metrics |
| TB loss > TF loss by <0.5 | 📉 TinyBrain slightly worse |
| TB loss > TF loss by >1.0 | ❌ Architecture needs iteration |
| γ grows from 0 to >0.05 | ✅ ThinkingStep is activating |
| gate grows from 0 to >0.05 | ✅ Memory is contributing |
| mem_util > 50% | ✅ Memory slots being used |
| mem_util < 10% | ⚠️ Memory mostly unused |

## Files To Save From RunPod

```bash
# Download these after training:
checkpoints/tinybrain_best.pt     # Best model
novacore/experiments/results/*.json  # All results