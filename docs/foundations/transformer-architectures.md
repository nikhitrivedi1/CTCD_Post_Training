# Transformer Architectures

A deep dive into transformer internals: from neurons to attention to complete architectures.

---

## Table of Contents

1. [Foundation: Neural Networks](#foundation-neural-networks)
2. [The Attention Mechanism](#the-attention-mechanism)
3. [Multi-Head Attention](#multi-head-attention)
4. [The Transformer Block](#the-transformer-block)
5. [Positional Encoding](#positional-encoding)
6. [The Three Architectures](#the-three-architectures)
7. [Architecture Comparison](#architecture-comparison)
8. [Practical Considerations](#practical-considerations)
9. [References](#references)

---

## Foundation: Neural Networks

Before understanding transformers, we need to understand the building blocks.

### The Perceptron (Single Neuron)

A single neuron computes a weighted sum plus bias, then applies an activation:

```
Inputs:     x₁, x₂, x₃, ..., xₙ
Weights:    w₁, w₂, w₃, ..., wₙ
Bias:       b

Computation:
    z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b  (weighted sum)
    y = σ(z)                            (activation)

         x₁ ──(w₁)──┐
         x₂ ──(w₂)──┼──[Σ + b]──[σ]──→ y
         x₃ ──(w₃)──┘
```

### Multi-Layer Perceptron (MLP)

An MLP stacks multiple layers of neurons:

```
    Input        Hidden        Output
    Layer        Layer         Layer

      ●            ●             ●
     (x₁)                       (y₁)

      ●            ●             ●
     (x₂)                       (y₂)

      ●            ●             ●
     (x₃)                       (y₃)

    (3 nodes)    (3 nodes)    (3 nodes)

Fully connected: every node connects to all nodes in next layer.
```

**Matrix Form:**
```
Hidden = Activation(Input × W₁ + b₁)
Output = Activation(Hidden × W₂ + b₂)
```

### Common Activation Functions

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| **ReLU** | max(0, x) | [0, ∞) | Hidden layers (default) |
| **GELU** | x × Φ(x) | (-∞, ∞) | Transformers (smoother) |
| **Sigmoid** | 1/(1+e⁻ˣ) | (0, 1) | Binary output, gates |
| **Tanh** | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | Centered activation |
| **Softmax** | eˣⁱ/Σeˣʲ | (0, 1), sum=1 | Classification output |
| **SiLU/Swish** | x × sigmoid(x) | (-∞, ∞) | Modern transformers |

### The Feed-Forward Network (FFN)

Transformers use a specific MLP pattern called the Feed-Forward Network:

```
FFN(x) = Activation(x × W₁ + b₁) × W₂ + b₂

Dimensions:
    Input:  x ∈ ℝᵈ                (d = hidden_size, e.g., 768)
    W₁:     ℝᵈ → ℝ⁴ᵈ             (expand 4x)
    W₂:     ℝ⁴ᵈ → ℝᵈ             (contract back)

Example (BERT-base):
    Input:  768 dimensions
    Hidden: 3072 dimensions (4 × 768)
    Output: 768 dimensions
```

**Why expand then contract?**
- Expansion creates a higher-dimensional space for complex transformations
- Contraction preserves the residual connection dimension
- The 4x ratio is empirically effective (though modern models vary this)

---

## The Attention Mechanism

Attention is the core innovation that makes transformers work.

### The Intuition

**Problem:** In sequence processing, how does each position "see" other positions?

**RNN solution:** Process sequentially, carry hidden state forward
- Limitation: Information decays over distance; can't parallelize

**Attention solution:** Each position directly queries all other positions
- Advantage: O(1) path between any two positions; fully parallel

### Query, Key, Value (QKV)

Attention uses three projections of the input:

```
Query (Q): "What am I looking for?"
Key (K):   "What do I contain?"
Value (V): "What information do I provide?"

For input X ∈ ℝⁿˣᵈ (n tokens, d dimensions):
    Q = X × Wq    (ℝⁿˣᵈ × ℝᵈˣᵈᵏ = ℝⁿˣᵈᵏ)
    K = X × Wk    (ℝⁿˣᵈ × ℝᵈˣᵈᵏ = ℝⁿˣᵈᵏ)
    V = X × Wv    (ℝⁿˣᵈ × ℝᵈˣᵈᵛ = ℝⁿˣᵈᵛ)
```

### Scaled Dot-Product Attention

The attention mechanism computes:

```
Attention(Q, K, V) = softmax(Q × Kᵀ / √dₖ) × V

Step by step:
    1. Q × Kᵀ           → Attention scores (ℝⁿˣⁿ)
    2. ÷ √dₖ            → Scale (prevents softmax saturation)
    3. softmax          → Attention weights (rows sum to 1)
    4. × V              → Weighted combination of values
```

**Visual representation:**

```
Token:     [The]  [cat]  [sat]  [on]  [mat]
              ↓      ↓      ↓     ↓      ↓
           ┌──────────────────────────────┐
           │  Compute Q, K, V for each    │
           └──────────────────────────────┘
              ↓      ↓      ↓     ↓      ↓
           Q: q₁    q₂    q₃    q₄    q₅
           K: k₁    k₂    k₃    k₄    k₅
           V: v₁    v₂    v₃    v₄    v₅
              ↓      ↓      ↓     ↓      ↓
           ┌──────────────────────────────┐
           │     Attention Scores         │
           │   (each q attends to all k)  │
           │                              │
           │        k₁  k₂  k₃  k₄  k₅    │
           │   q₁ [.1  .3  .2  .1  .3]   │ → weighted sum of v's
           │   q₂ [.2  .4  .2  .1  .1]   │ → weighted sum of v's
           │   q₃ [.1  .2  .5  .1  .1]   │ → weighted sum of v's
           │   q₄ [.1  .1  .2  .4  .2]   │ → weighted sum of v's
           │   q₅ [.2  .3  .1  .2  .2]   │ → weighted sum of v's
           └──────────────────────────────┘
```

### Why Scale by √dₖ?

Without scaling, dot products grow with dimension:

```
If q, k are random vectors with variance 1:
    E[q · k] = 0
    Var[q · k] = dₖ

As dₖ grows:
    - Dot products have larger variance
    - Softmax becomes sharper (near one-hot)
    - Gradients vanish

Scaling by √dₖ normalizes variance to 1.
```

### Self-Attention vs Cross-Attention

| Type | Q source | K, V source | Use case |
|------|----------|-------------|----------|
| **Self-attention** | X | X | Encoder, decoder self |
| **Cross-attention** | Decoder | Encoder | Encoder-decoder models |

```
Self-Attention (Encoder):
    Q, K, V all from same sequence X
    Each token attends to all tokens in X

Cross-Attention (Decoder attending to Encoder):
    Q from decoder hidden states
    K, V from encoder outputs
    Decoder tokens attend to encoder tokens
```

---

## Multi-Head Attention

Instead of one attention function, use multiple "heads" in parallel.

### Why Multiple Heads?

Single attention can only focus on one type of relationship. Multiple heads can capture:
- Syntactic relationships (subject-verb)
- Semantic relationships (synonyms)
- Positional patterns (adjacent words)
- Long-range dependencies

### Multi-Head Computation

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₕ) × Wₒ

Where:
    headᵢ = Attention(Q × Wqᵢ, K × Wkᵢ, V × Wvᵢ)

Dimensions (BERT-base example):
    d_model = 768 (hidden size)
    h = 12 (number of heads)
    d_k = d_v = d_model / h = 64 (per-head dimension)
```

**Visual:**

```
Input X (768-dim)
    │
    ├──→ Head 1: Wq₁, Wk₁, Wv₁ (768→64) → Attention → (64-dim)
    ├──→ Head 2: Wq₂, Wk₂, Wv₂ (768→64) → Attention → (64-dim)
    ├──→ ...
    └──→ Head 12: Wq₁₂, Wk₁₂, Wv₁₂ (768→64) → Attention → (64-dim)
                                                          │
                                    Concat ←──────────────┘
                                      │ (768-dim)
                                      ↓
                                    Wₒ (768→768)
                                      │
                                      ↓
                                Output (768-dim)
```

### Parameter Count

For multi-head attention:

```
Wq, Wk, Wv: 3 × (d_model × d_model) = 3 × 768 × 768 = 1,769,472
Wₒ:         d_model × d_model = 768 × 768 = 589,824
Total:      2,359,296 parameters per attention layer
```

---

## The Transformer Block

A transformer block combines attention with feed-forward processing.

### Block Structure

```
┌───────────────────────────────────────────┐
│           Transformer Block               │
├───────────────────────────────────────────┤
│                                           │
│   Input ───────────┬──────────────────    │
│      │             │                      │
│      ↓             │ (residual)           │
│  ┌──────────┐      │                      │
│  │LayerNorm │      │                      │
│  └──────────┘      │                      │
│      │             │                      │
│      ↓             │                      │
│  ┌──────────┐      │                      │
│  │Multi-Head│      │                      │
│  │Attention │      │                      │
│  └──────────┘      │                      │
│      │             │                      │
│      ↓             │                      │
│  ┌──────────┐      │                      │
│  │ Dropout  │      │                      │
│  └──────────┘      │                      │
│      │             │                      │
│      └─────(+)─────┘                      │
│            │                              │
│   ┌────────┴────────┬─────────────────    │
│   │                 │                     │
│   ↓                 │ (residual)          │
│  ┌──────────┐       │                     │
│  │LayerNorm │       │                     │
│  └──────────┘       │                     │
│   │                 │                     │
│   ↓                 │                     │
│  ┌──────────┐       │                     │
│  │   FFN    │       │                     │
│  └──────────┘       │                     │
│   │                 │                     │
│   ↓                 │                     │
│  ┌──────────┐       │                     │
│  │ Dropout  │       │                     │
│  └──────────┘       │                     │
│   │                 │                     │
│   └─────(+)─────────┘                     │
│         │                                 │
│      Output                               │
│                                           │
└───────────────────────────────────────────┘
```

### Pre-Norm vs Post-Norm

Two common arrangements:

```
Post-Norm (Original Transformer):          Pre-Norm (GPT-2, modern):
    x → Attention → Add → LN →                x → LN → Attention → Add →
      → FFN → Add → LN → output                 → LN → FFN → Add → output

Post-Norm: LayerNorm after residual        Pre-Norm: LayerNorm before sublayer
- Original paper                            - More stable training
- Needs careful initialization              - Allows higher learning rates
- Warmup critical                           - Default for large models
```

### Residual Connections

Residual (skip) connections are critical:

```
output = sublayer(x) + x

Benefits:
1. Gradient flow: Gradients pass through unchanged
2. Identity default: Network can learn identity easily
3. Deep training: Enables 100+ layer networks
```

Without residuals, gradients must flow through every transformation, leading to vanishing/exploding gradients.

### Layer Normalization

Normalizes activations across the feature dimension:

```
LayerNorm(x) = γ × (x - μ) / √(σ² + ε) + β

Where:
    μ = mean across features (per token)
    σ² = variance across features (per token)
    γ, β = learned scale and shift (per feature)
    ε = small constant for numerical stability

Difference from BatchNorm:
    BatchNorm: Normalize across batch dimension
    LayerNorm: Normalize across feature dimension

    LayerNorm is batch-size independent → better for variable sequences
```

---

## Positional Encoding

Attention is permutation-invariant—it doesn't know token order. Positional encoding adds this information.

### Sinusoidal Encoding (Original)

```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Where:
    pos = position in sequence (0, 1, 2, ...)
    i = dimension index (0, 1, ..., d/2)
    d = model dimension

Properties:
    - Unique encoding for each position
    - Relative positions can be computed via linear transformation
    - Extrapolates to longer sequences than seen in training
```

**Visualization:**

```
Position    Dim 0   Dim 1   Dim 2   Dim 3   ...
    0       sin(0)  cos(0)  sin(0)  cos(0)  ...
    1       sin(1)  cos(1)  sin(.)  cos(.)  ...
    2       sin(2)  cos(2)  sin(.)  cos(.)  ...
    ...

Lower dimensions: High frequency (change rapidly with position)
Higher dimensions: Low frequency (change slowly)
```

### Learned Positional Embeddings

Modern models often learn position embeddings:

```
PE = Embedding(position_ids)  # Learned lookup table

Properties:
    - More flexible than sinusoidal
    - Limited to max sequence length seen in training
    - Used by BERT, GPT-2
```

### Rotary Position Embeddings (RoPE)

Used by Llama, modern models:

```
RoPE encodes position by rotating Q and K vectors:
    q'_m = R_m × q
    k'_n = R_n × k

Where R is a rotation matrix based on position.

Properties:
    - Relative position encoded in dot product
    - Better extrapolation to longer sequences
    - No added parameters
```

### Comparison

| Method | Pros | Cons | Used by |
|--------|------|------|---------|
| **Sinusoidal** | No params, extrapolates | Less flexible | Original Transformer |
| **Learned** | Flexible, simple | Fixed max length | BERT, GPT-2 |
| **RoPE** | Relative positions, extrapolates | More complex | Llama, Mistral |
| **ALiBi** | Simple, extrapolates | Attention bias only | BLOOM |

---

## The Three Architectures

Transformers come in three main variants based on attention patterns.

### Attention Masking

The key difference is **what each token can attend to**:

```
Bidirectional (Encoder):        Causal (Decoder):
    Can see ALL tokens              Can only see PAST tokens

    Attention Matrix:               Attention Matrix:
    [1 1 1 1 1]                     [1 0 0 0 0]
    [1 1 1 1 1]                     [1 1 0 0 0]
    [1 1 1 1 1]                     [1 1 1 0 0]
    [1 1 1 1 1]                     [1 1 1 1 0]
    [1 1 1 1 1]                     [1 1 1 1 1]

    1 = can attend                  Upper triangle masked (-∞)
    (no mask)                       Prevents "seeing future"
```

### Encoder-Only (Bidirectional)

**Architecture:**

```
Input Tokens: [CLS] The cat sat [SEP]
       │
       ↓
  ┌─────────────┐
  │  Embedding  │  (token + position + segment)
  └─────────────┘
       │
       ↓
  ┌─────────────┐
  │  Encoder    │  × N layers
  │  Block      │  (bidirectional self-attention)
  └─────────────┘
       │
       ↓
  Output Embeddings (one per token)
       │
       ↓
  [Task Head]  → Classification, Token labeling, etc.
```

**Characteristics:**
- Each token sees all other tokens (bidirectional)
- Best for understanding/classification tasks
- Cannot generate text autoregressively
- Examples: BERT, RoBERTa, ALBERT, DistilBERT

**Use Cases:**
- Text classification (use [CLS] token)
- Named entity recognition (use all tokens)
- Question answering (extractive)
- Sentence embeddings

### Decoder-Only (Causal/Autoregressive)

**Architecture:**

```
Input Tokens: The cat sat on
       │
       ↓
  ┌─────────────┐
  │  Embedding  │  (token + position)
  └─────────────┘
       │
       ↓
  ┌─────────────┐
  │  Decoder    │  × N layers
  │  Block      │  (causal self-attention)
  └─────────────┘
       │
       ↓
  Output Embeddings (one per token)
       │
       ↓
  ┌─────────────┐
  │   LM Head   │  → Predict next token
  └─────────────┘
       │
       ↓
  "the" → predict: mat, floor, bed, ...
```

**Causal Attention Detail:**

```
Query at position 3 ("sat"):

    Tokens:  [The]  [cat]  [sat]  [on]   [mat]
    Mask:      1      1      1      0      0
                                   ↑      ↑
                              Cannot see future

    Attention only over: The, cat, sat
    Prediction: What comes after "sat"?
```

**Characteristics:**
- Each token only sees previous tokens (causal mask)
- Trained to predict next token
- Can generate text autoregressively
- Examples: GPT-2, GPT-3, Llama, Mistral

**Use Cases:**
- Text generation
- Code completion
- Chat/dialogue
- Any autoregressive task

### Encoder-Decoder (Seq2Seq)

**Architecture:**

```
Input: "The cat sat"                    Output: "Die Katze saß"

        ENCODER                                 DECODER
    (Bidirectional)                            (Causal)

  [The] [cat] [sat]                      [<s>] [Die] [Katze]
     │     │     │                          │     │      │
     ↓     ↓     ↓                          ↓     ↓      ↓
  ┌─────────────────┐                   ┌─────────────────┐
  │    Embedding    │                   │    Embedding    │
  └─────────────────┘                   └─────────────────┘
     │     │     │                          │     │      │
     ↓     ↓     ↓                          ↓     ↓      ↓
  ┌─────────────────┐                   ┌─────────────────┐
  │  Encoder Block  │                   │Causal Self-Attn │
  │  (Self-Attn +   │                   │    (masked)     │
  │      FFN)       │                   └─────────────────┘
  └─────────────────┘                       │     │      │
     │     │     │                          ↓     ↓      ↓
     │     │     │                      ┌─────────────────┐
     │     │     │    ──────────────→   │ Cross-Attention │
     │     │     │    K, V from encoder │  Q: decoder     │
     ↓     ↓     ↓                      └─────────────────┘
  Encoder Output                            │     │      │
    (memory)                                ↓     ↓      ↓
                                        ┌─────────────────┐
                                        │       FFN       │
                                        └─────────────────┘
                                            │     │      │
                                            ↓     ↓      ↓
                                        [Die] [Katze] [saß]
```

**Decoder Block (with Cross-Attention):**

```
┌─────────────────────────────────────┐
│      Encoder-Decoder Block          │
├─────────────────────────────────────┤
│                                     │
│  Input ─────────┬───────────────    │
│     │           │                   │
│     ↓           │ (residual)        │
│  ┌────────┐     │                   │
│  │ Masked │     │                   │
│  │Self-Att│ ← Causal mask           │
│  └────────┘     │                   │
│     │           │                   │
│     └───(+)─────┘                   │
│         │                           │
│  ┌──────┴───────┬───────────────    │
│  │              │                   │
│  ↓              │ (residual)        │
│  ┌────────┐     │                   │
│  │ Cross- │     │                   │
│  │  Attn  │ ← K,V from encoder      │
│  └────────┘     │                   │
│  │              │                   │
│  └───(+)────────┘                   │
│      │                              │
│  ┌───┴──────────┬───────────────    │
│  │              │                   │
│  ↓              │ (residual)        │
│  ┌────────┐     │                   │
│  │  FFN   │     │                   │
│  └────────┘     │                   │
│  │              │                   │
│  └───(+)────────┘                   │
│      │                              │
│   Output                            │
│                                     │
└─────────────────────────────────────┘
```

**Characteristics:**
- Encoder: Bidirectional attention on input
- Decoder: Causal self-attention + cross-attention to encoder
- Best for sequence-to-sequence tasks
- Examples: T5, BART, mBART, MarianMT

**Use Cases:**
- Machine translation
- Summarization
- Question answering (generative)
- Any input→output transformation

---

## Architecture Comparison

### Structural Differences

| Aspect | Encoder-Only | Decoder-Only | Encoder-Decoder |
|--------|--------------|--------------|-----------------|
| **Self-attention** | Bidirectional | Causal (masked) | Encoder: bidir, Decoder: causal |
| **Cross-attention** | No | No | Yes (decoder→encoder) |
| **Output** | Per-token embeddings | Next token prediction | Sequence generation |
| **Training** | MLM, NSP | Next token prediction | Seq2seq |
| **Generation** | Cannot | Autoregressive | Autoregressive |

### Attention Patterns Visualized

```
ENCODER-ONLY (BERT)                      DECODER-ONLY (GPT)
───────────────────                      ──────────────────
"The cat sat on mat"                     "The cat sat on mat"

    T c s o m                                T c s o m
  T[● ● ● ● ●]                            T[● · · · ·]
  c[● ● ● ● ●]                            c[● ● · · ·]
  s[● ● ● ● ●]                            s[● ● ● · ·]
  o[● ● ● ● ●]                            o[● ● ● ● ·]
  m[● ● ● ● ●]                            m[● ● ● ● ●]

All tokens attend                        Each token attends only
to all tokens                            to previous (lower triangle)


ENCODER-DECODER (T5)
────────────────────
Encoder: Full attention | Decoder: Causal + cross-attention

Decoder Self-Attention               Decoder Cross-Attention
(to decoder tokens)                  (to encoder tokens)

    D1 D2 D3                             E1 E2 E3 E4
 D1[●  ·  · ]                         D1[●  ●  ●  ● ]
 D2[●  ●  · ]                         D2[●  ●  ●  ● ]
 D3[●  ●  ● ]                         D3[●  ●  ●  ● ]

Causal (lower triangle)              Full attention to encoder
```

### Parameter Distribution

For a typical transformer (BERT-base scale):

```
Component                   Parameters      % of Total
─────────────────────────────────────────────────────
Embeddings
  Token embedding           23.4M           21%
  Position embedding        0.6M            <1%
  (Segment embedding)       0.02M           <1%

Per Layer (×12)
  Multi-head attention      2.4M            22%
  Feed-forward network      4.7M            43%
  Layer norms               0.003M          <1%

Output head                 varies          varies
─────────────────────────────────────────────────────
Total (BERT-base)           ~110M           100%
```

### Computational Complexity

| Operation | Time Complexity | Memory Complexity |
|-----------|-----------------|-------------------|
| Self-attention | O(n² × d) | O(n² + n × d) |
| FFN | O(n × d²) | O(n × d) |
| Total per layer | O(n² × d + n × d²) | O(n² + n × d) |

Where n = sequence length, d = hidden dimension

**Bottleneck:** Self-attention is quadratic in sequence length
- n=512: 262K attention pairs
- n=2048: 4.2M attention pairs
- n=8192: 67M attention pairs

This motivates efficient attention variants (sparse, linear, flash attention).

---

## Practical Considerations

### Model Selection Guide

```
Task: Text Classification
    → Encoder-only (BERT, RoBERTa)
    → Use [CLS] token or pool all tokens

Task: Text Generation
    → Decoder-only (GPT, Llama)
    → Autoregressive generation

Task: Translation/Summarization
    → Encoder-decoder (T5, BART) OR
    → Decoder-only with prompt (modern approach)

Task: Information Retrieval
    → Encoder-only for embeddings
    → Or decoder-only with mean pooling
```

### Common Model Configurations

| Model | Type | Layers | Hidden | Heads | FFN | Params |
|-------|------|--------|--------|-------|-----|--------|
| BERT-base | Enc | 12 | 768 | 12 | 3072 | 110M |
| BERT-large | Enc | 24 | 1024 | 16 | 4096 | 340M |
| GPT-2 | Dec | 12 | 768 | 12 | 3072 | 117M |
| GPT-2-XL | Dec | 48 | 1600 | 25 | 6400 | 1.5B |
| T5-base | Enc-Dec | 12+12 | 768 | 12 | 3072 | 220M |
| Llama-3.2-1B | Dec | 16 | 2048 | 32 | 8192 | 1B |
| Llama-3.2-3B | Dec | 28 | 3072 | 24 | 8192 | 3B |

### Distillation Considerations

When distilling transformers, architecture differences matter:

**Same Architecture (easier):**
```
BERT-base → DistilBERT
- Both encoder-only
- Student has fewer layers (12 → 6)
- Hidden size preserved (768 → 768)
- Feature distillation: Direct layer mapping
```

**Different Sizes (moderate):**
```
Llama-3B → Llama-1B
- Both decoder-only
- Different hidden sizes (3072 → 2048)
- Different layer counts (28 → 16)
- Feature distillation: Requires projection layer
```

**Different Architectures (hard):**
```
T5 → GPT
- Encoder-decoder vs decoder-only
- Cross-attention doesn't exist in student
- Response distillation only (logit matching)
```

---

## References

### Foundational Papers

1. **Attention Is All You Need**
   Vaswani et al. (2017)
   - Original transformer architecture
   - Encoder-decoder for translation

2. **BERT: Pre-training of Deep Bidirectional Transformers**
   Devlin et al. (2019)
   - Encoder-only architecture
   - Masked language modeling

3. **Language Models are Unsupervised Multitask Learners**
   Radford et al. (2019) - GPT-2
   - Decoder-only architecture
   - Autoregressive pretraining

4. **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**
   Raffel et al. (2020) - T5
   - Encoder-decoder at scale
   - Text-to-text framework

### Architecture Innovations

5. **RoFormer: Enhanced Transformer with Rotary Position Embedding**
   Su et al. (2021)
   - Rotary position embeddings (RoPE)

6. **GLU Variants Improve Transformer**
   Shazeer (2020)
   - SwiGLU, GeGLU activations

7. **FlashAttention: Fast and Memory-Efficient Exact Attention**
   Dao et al. (2022)
   - IO-aware attention computation

### Scaling and Efficiency

8. **Scaling Laws for Neural Language Models**
   Kaplan et al. (2020)
   - Relationship between compute, data, parameters

9. **Training Compute-Optimal Large Language Models**
   Hoffmann et al. (2022) - Chinchilla
   - Optimal scaling ratios

10. **Llama 2: Open Foundation and Fine-Tuned Chat Models**
    Touvron et al. (2023)
    - Modern decoder-only design choices

---

## See Also

- [Tasks, Models, Datasets](tasks-models-datasets.md) - ML paradigms and task types
- [Paper 1: Ablation Study](../paper/paper-1-ablation.md) - Distillation findings
- [Paper 2: FAR Study](../paper/paper-2-far.md) - Multi-signal distillation
