# Nova Architecture — Computation Specification

> Phase 0: Complete mathematical definition before any scaling.
> 20-30 pages of architecture theory before 5M implementation.

---

## 0. Notation

```
d  = hidden_size          (1024 → 2048)
m  = memory_slots         (64 → 128)
n  = sequence_length      (variable)
T  = thinking_steps       (variable, 1-32)
K  = num_cells            (6 → 12)
V  = vocab_size           (152064 for Qwen)

x ∈ ℝ^d          — token state vector
M ∈ ℝ^{m×d}     — memory matrix (fixed size, not growing)
W ∈ ℝ^{d×d}     — learned projection matrix
σ = sigmoid
⊙ = element-wise multiply
‖·‖ = L2 norm
```

---

## 1. Brain Cell — What Will It Do?

### Core Hypothesis

**Iterative dimension-wise correction is more compute-efficient than token-pair mixing for next-token prediction.**

### The Standard (Transformer)

```
Attention:  A = softmax(QK^T/√d)V     → mixes token information
MLP:        H = σ(W_g·A) ⊙ (W_u·A)    → processes each token
```

Problem: `O(n²·d)` compute for attention — 90% of inference FLOPs at long sequences.

### Nova's Approach

```
Observation:  O = W_o · LayerNorm(x)                   [ℝ^d]
Compare:      C = tanh(W_c · O + b_c)                   [ℝ^d]
Error:        E = SiLU(W_e1 · O) ⊙ (W_e2 · O)          [ℝ^d]
Correction:   Δ = tanh(γ · C ⊙ E)                       [ℝ^d]
Update:       x_{t+1} = x_t + Δ                         [ℝ^d]
```

No quadratic attention. No token mixing. Each dimension independently:
1. **Observes** its current value
2. **Compares** it to a learned target
3. **Predicts** what's missing (error)
4. **Corrects** the state

Complexity: `O(K · n · d²)` vs Transformer's `O(L · n² · d)`

### Why This Could Work

Next-token prediction is fundamentally *local*: "given tokens 1..t, predict token t+1". The relationship between token t and token t+1 is mostly captured by position, not by long-range attention. The Brain Cell exploits this by operating per-dimension rather than per-token-pair.

### Critical Question

> Does `tanh(W_c · O)` compute something fundamentally different from `softmax(QK^T)V`?

Answer: **Yes**. Attention computes a **weighted sum over tokens**. The compare gate computes a **dimension-wise gated activation**. The former is relational (token A vs token B), the latter is functional (what to change in this dimension).

---

## 2. Memory — How Will It Store Information?

### Core Hypothesis

**Fixed-size compressed memory can match or exceed unbounded KV cache at a fraction of the memory cost.**

### Memory Structure

```
M ∈ ℝ^{m×d}     — m learned memory slots, each d-dimensional
```

| Property | KV Cache | LearnedMemory |
|----------|----------|--------------|
| Size | O(n·d) — grows with sequence | O(m·d) — fixed (e.g. 64 × 1024) |
| Access | Position-based (cache index) | Content-based (attention over slots) |
| Retention | Perfect until evicted | Learned (importance decay) |
| Cost at 4K seq | ~256 MB for 512d | ~0.5 MB for 64 slots |

### Read Operation

```
q = W_read · LayerNorm(x_t)                    [query from state]
α = softmax(q · M^T / √d)                      [attention over slots]
r = Σ_i α_i · M_i                               [weighted read]
```

This **is** attention — but over `m=64` learned slots, not `n=4096` tokens.

### Write Operation

```
v = W_value · x_t                               [value from state]
g_write = σ(W_write · x_t)                      [per-slot write gate]
g_erase = σ(W_erase · x_t)                      [per-slot erase gate]

M_i ← M_i · (1 - g_erase_i) + g_write_i · v     [erase-then-add]
```

### Compression

```
I = sigmoid(W_imp · M)                          [importance per slot]
Training:   M ← M ⊙ (1 - (1-I)·η)               [soft decay]
Inference:  M ← M ⊙ (I > τ)                    [hard prune, τ=0.1]
```

### Critical Question

> Can 64 slots retain enough information for sequences of 4096+ tokens?

Test: Needle-in-Haystack at seq_len=4096. If retrieval accuracy > 90%, memory works. If not, increase m or change addressing mechanism.

---

## 3. State — How Will It Update?

### Core Hypothesis

**A bounded, learnable-gated recurrent update is more stable and trainable than an unbounded additive update.**

### State Update Equation

```
x_{t+1} = x_t + tanh(γ) · Δ_t + tanh(α) · r_t
```

Where:
- `γ`: learnable scaler for thinking correction (init=0)
- `α`: learnable scaler for memory readout (init=0)
- `Δ_t`: thinking correction (bounded by tanh)
- `r_t`: memory readout (bounded by output gate)

### Stability Guarantee

At initialization (γ=0, α=0):
```
x_{t+1} = x_t
```
The model starts as **identity mapping**. No gradient explosion. No hidden state collapse.

During training, γ and α learn to open up:
```
γ > 0  → thinking corrections are applied
α > 0  → memory readout influences state
```

### Critical Questions

> Does γ ever grow beyond ~0.1?

From experiments: gamma reaches ~0.006 after 200 steps on random data. On real data (TinyStories), it should reach 0.1-0.5.

> What if γ stays at 0?

The architecture becomes a linear embedding → LM head, which cannot learn language. γ growing is a necessary condition for architecture validity.

---

## 4. Learning Rule — What Will It Be?

### Core Hypothesis

**Cross-entropy + auxiliary losses for confidence and step count can train a dynamic-thinking architecture end-to-end.**

### Loss Function

```
L = L_lm + λ_m · L_memory + λ_c · L_confidence + λ_s · L_steps
```

| Component | Equation | λ |
|-----------|----------|---|
| Language modeling | `-Σ log P(y_t | x_{<t})` | 1.0 |
| Memory regularization | `(1/m) · Σ||M_i||²` | 0.001 |
| Confidence penalty | `(1/K·T) · Σ(1 - c_t)` | 0.01 |
| Step penalty | `(T_cell / T_max)²` | 0.1 |

### Gradient Flow

```
L_lm → LM Head → Output MLP → SelfCorrection → Cell N → ... → Cell 1 → Embedding
                                                          ↓
                                                     ThinkingStep params
                                                          ↓
                                                     γ (gate), W_o, W_c, W_e

L_memory → Memory slots (M) → Memory projections (W_read, W_write, etc.)
L_confidence → Confidence Gate params (W_conf, step_embed)
L_steps → (gradient through Gumbel-sigmoid halt decision)
```

### Critical Question

> Can gradients flow through the dynamic halt decision?

Yes — Gumbel-sigmoid provides `∂h/∂θ` through the binary halt gate. During training, the model learns: "if I halt too early, L_lm is high; if I halt too late, L_steps is high."

---

## 5. Output — How Will It Generate?

### Core Hypothesis

**Standard softmax output with self-correction can match or exceed LM head-only output.**

### Forward Pass

```
1. Embed(input_ids)                                  [V → d]
2. For each cell k = 1..K:
     For step t = 1..T_k:
       x = ThinkingStep(x)                            [§1]
       r, M = Memory(x, M)                            [§2]
       x = x + tanh(α) · r                            [§3]
       if Confidence(x, x₀, t) > γ: break             [dynamic halt]
3. x = SelfCorrection(x)                              [verify + refine]
4. x = OutputMLP(x)                                   [d → d_ff → d]
5. logits = LMHead(x)                                 [d → V]
6. P(next_token) = softmax(logits)
```

### Self-Correction

```
For each correction step:
  v = σ(W_verify · LN(x))              [verification score: 0-1]
  r = SiLU(W_refine · LN(x))           [refinement signal]
  g = σ(W_gate · x) ⊙ (1 - v)          [apply refinement where uncertain]
  x = x + r ⊙ g                         [corrected state]
```

### Inference Generation

```
For each new token:
  output = model(prompt)
  next_token = sample(output.logits[-1])
  prompt = [prompt; next_token]
  states = output.memory_states         [carry over memory]
```

### Critical Question

> Can SelfCorrection improve output without being an extra MLP that adds no value?

Test: Ablation — remove SelfCorrection. If perplexity doesn't change, remove it permanently.

---

## 6. Model — When Will It Stop?

### Core Hypothesis

**Confidence-based dynamic stopping can match fixed-depth models at lower average compute.**

### Confidence Definition

```
c_t = σ(W_conf · [x_t; x_t - x_0] + step_embed + b_conf)
```

Where:
- `x_t`: current state after t thinking steps
- `x_0`: initial state (before thinking)
- `[x_t; x_t - x_0]`: concatenation (2d dimensions)
- `step_embed`: learned position encoding
- Result: c_t ∈ [0, 1]

### Halting Decision

```
Training:   h_t = gumbel_sigmoid(logit(c_t), τ=1.0)     [differentiable]
Inference:  h_t = 1 if c_t ≥ 0.85 else 0                [deterministic]
```

### Expected Behavior

| Input Difficulty | Thinking Steps | Relative Compute |
|-----------------|---------------|------------------|
| Easy (common patterns) | 1-3 | 0.1× |
| Medium (familiar) | 4-8 | 0.25× |
| Hard (novel reasoning) | 8-16 | 0.5× |
| Very Hard | 16-32 | 1.0× |

### Critical Question

> After training, does the model actually use fewer steps for easy inputs?

Test: Sort validation set by difficulty (loss). Plot thinking steps vs difficulty. If correlation is positive, dynamic compute works.

---

## 7. Compute — How Will It Scale?

### Core Hypothesis

**TinyBrain scales favorably because its O(n·d²) complexity dominates over O(n²·d) attention only at very long sequences or large d.**

### FLOPs Comparison

| Operation | Transformer | TinyBrain | Ratio at n=4096, d=1024 |
|-----------|-------------|-----------|------------------------|
| Token mixing | `2·n²·d` | `0` | → 0× |
| Per-token processing | `8·n·d²` | `5·K·T·d²` | → depends on K·T |
| Memory access | `n·d` | `2·m·d + n·m` | → 0.01× |

### Where TinyBrain Wins

| Regime | Transformer | TinyBrain | Winner |
|--------|-------------|-----------|--------|
| Short context (n=512) | O(n²·d) dominant | O(n·d²) dominant | Depends on T |
| Long context (n=4096) | O(n²·d) = 16B FLOPs | O(n·d²) = 0.5B FLOPs | TinyBrain |
| Batch inference | KV cache O(n·dL) | Memory O(m·dK) | TinyBrain |

### Scaling Law Prediction

```
Loss(N, D) = a·N^(-b) + c·D^(-d) + e

Where:
  N = number of parameters
  D = number of tokens
  a,b,c,d,e = fitted constants
```

Expected: TinyBrain's exponent b should be ≥ Transformer's b (same or better scaling with parameters).

---

## 8. Transformer से मूल अंतर क्या है?

| Property | Transformer | TinyBrain (Nova) |
|----------|-----------|-----------------|
| **Core operation** | Token-pair similarity (QK^T) | Dimension-wise gating (tanh) |
| **Complexity** | O(n²·d) | O(K·T·n·d) |
| **Memory** | KV cache: O(n·dL) | LearnedMemory: O(m·dK) |
| **Depth** | Fixed (same for all inputs) | Dynamic (varies per token) |
| **Position encoding** | RoPE / ALiBi (explicit) | Step embedding (implicit) |
| **Residual** | Standard | Learnable-gated (γ, α) |
| **Token mixing** | Across all positions | None (per-dimension only) |
| **Information retention** | Via attention to past tokens | Via explicit memory write |
| **Correction mechanism** | None (feed-forward only) | Explicit error prediction |
| **Training stability** | Well-understood | Requires γ=0 initialization |

### Fundamental Difference

**Transformer** is a *relational* architecture: it computes **which token relates to which** and mixes them.

**TinyBrain** is a *functional* architecture: it computes **what correction to apply** to each dimension based on its current state.

Both can model next-token prediction, but they make different tradeoffs:
- Transformer excels at tasks requiring explicit token relationships (coreference resolution, translation)
- TinyBrain should excel at tasks requiring iterative refinement (reasoning, generation, planning)

### Future Work

1. Can TinyBrain and Transformer be combined? (e.g., TinyBrain for generation, sparse attention for retrieval)
2. Does TinyBrain scale better to long contexts (256K+) than linear attention variants?
3. Can TinyBrain be extended to multi-modal by adding modality-specific "observations"?

---

*This document defines the architecture completely. Next: 5M parameter proof-of-concept.*