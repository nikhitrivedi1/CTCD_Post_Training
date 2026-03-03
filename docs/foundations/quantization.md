# Quantization

## What quantization does

A model is mostly matrix multiplications: `Y = X @ W`. Each weight `W` is stored as a float (FP32 = 32 bits, FP16 = 16 bits). Quantization represents these values in fewer bits.

**Number formats:**

| Format | Bits | Layout | Notes |
|---|---|---|---|
| FP32 | 32 | 1 sign + 8 exp + 23 mantissa | Full precision, huge range |
| FP16 | 16 | 1 sign + 5 exp + 10 mantissa | Less precise, smaller range |
| BF16 | 16 | 1 sign + 8 exp + 7 mantissa | Same range as FP32, less precise (Google's design) |
| FP8 | 8 | 1 sign + 4 exp + 3 mantissa | Very coarse but usable, needs Hopper+ hardware |
| INT8 | 8 | 8-bit integer + scale factor | Needs calibration |
| INT4 | 4 | 4-bit integer + scale factor | Very coarse, needs per-group scales |

## Core mechanics

### Linear quantization (INT8/INT4)

```
quantized_value = round(float_value / scale) + zero_point
dequantized_value = (quantized_value - zero_point) * scale
```

The scale/zero_point can be computed at different granularities:
- **Per-tensor:** one scale for the whole weight matrix (least accurate)
- **Per-channel:** one scale per output channel (good balance)
- **Per-group:** one scale per group of 128 values (GPTQ/AWQ use this, most accurate)

### FP8 quantization

No integer math. Just casts floats to a narrower float format. Simpler, but needs hardware support (Hopper/H200 has FP8 tensor cores).

## Touchpoints: Inference

A single transformer layer forward pass:

```
Input activations (X)  [batch, seq, hidden]
        |
        v
   QKV Projection:  Q,K,V = X @ W_q, X @ W_k, X @ W_v    <-- QUANTIZATION HERE (weights)
        |
        v
   Attention:  attn = softmax(Q @ K^T / sqrt(d)) @ V       <-- usually stays in FP16/BF16
        |
        v
   Output Projection:  out = attn @ W_o                     <-- QUANTIZATION HERE (weights)
        |
        v
   FFN Layer 1:  h = GELU(out @ W_up)                       <-- QUANTIZATION HERE
        |
        v
   FFN Layer 2:  output = h @ W_down                        <-- QUANTIZATION HERE
        |
        v
   Output activations [batch, seq, hidden]
```

Every `@` (matmul) is a touchpoint. With weight-only quantization (most common):
1. Weights stored in INT4/INT8/FP8 on GPU memory
2. Before each matmul, weights are **dequantized** to FP16/BF16
3. Matmul runs in FP16/BF16
4. Result stays in FP16/BF16

The win: step 1 means loading 2-4x fewer bytes from GPU memory. Since inference is **memory-bandwidth bound** (the GPU spends most time waiting for data, not computing), this directly translates to speedup.

**Weight-and-activation quantization (INT8 compute):**
- Both weights AND activations are quantized to INT8
- Matmul runs on INT8 tensor cores (2x throughput vs FP16)
- Requires calibration data to determine activation scales
- More complex, but faster when compute-bound

## Touchpoints: Training

Standard training does NOT quantize model weights. Here's why:

```
Forward:   loss = model(input)          # FP16/BF16 compute
Backward:  grads = loss.backward()      # FP16/BF16 gradients
Update:    W = W - lr * grad            # MUST be FP32
```

The update step is the problem. Gradients are tiny (1e-5 to 1e-8). In FP16, the smallest representable number is ~6e-8. So small gradient updates get **rounded to zero** and training stalls. FP32 can represent down to ~1e-38.

### Mixed precision training

This is what we use with `fp16: true` in our configs:

```
Master weights (FP32)  ----copy---->  FP16 weights
                                          |
                                     Forward pass (FP16) --> loss
                                          |
                                     Backward pass (FP16) --> FP16 gradients
                                          |
FP32 gradients  <----cast----       FP16 gradients
       |
  Optimizer step (FP32): W_fp32 = W_fp32 - lr * grad_fp32
       |
       ----copy---->  FP16 weights (for next iteration)
```

Memory: you need both FP32 master weights AND FP16 working copy. But activations (the big memory consumer during training) are all FP16, saving significant memory.

**Loss scaling:** FP16 gradients can underflow. The loss is scaled up (e.g., 1024x) before backward, then gradients are scaled back down before the optimizer step. PyTorch's `GradScaler` handles this automatically.

## Touchpoints: QLoRA (quantized training)

```
Base model weights: 4-bit (frozen, never updated)
LoRA adapters: FP16 (trainable, tiny - rank 8-64)

Forward:
  X @ dequantize(W_base_4bit) + X @ W_lora_A @ W_lora_B
  |                              |
  4bit->FP16 on the fly          Pure FP16, has gradients

Backward:
  Only W_lora_A, W_lora_B get gradients
  W_base stays in 4-bit, never updated
```

This is how you fine-tune a 7B model on a 24GB GPU. The base model takes ~3.5GB in 4-bit instead of ~14GB in FP16.

## Touchpoints: Distillation

```
Teacher (frozen):
  Can be quantized (FP8/INT8) since it only does forward passes
  No gradients needed, so precision loss is acceptable
  Example: olmOCR-2-7B-1025-FP8 loads as FP8, dequantizes to BF16 for compute

Student (training):
  Must stay in FP16/BF16 (or mixed precision with FP32 master)
  Needs full gradient precision for learning

Distillation loss:
  KL(student_logits / T, teacher_logits / T)
  Both logits computed in FP16/BF16
  Temperature T amplifies small differences, so precision matters here
```

### The olmOCR FP8 model specifically

When we load `allenai/olmOCR-2-7B-1025-FP8`:
1. `compressed-tensors` library reads FP8 weights from disk (~8GB vs ~16GB for BF16)
2. Each linear layer stores weights in FP8 + per-channel scales
3. On forward pass: `output = input @ dequantize(W_fp8, scale).to(bfloat16)`
4. The `torch_dtype=bfloat16` we pass tells it what precision to dequantize TO

The model is the same architecture, same number of parameters. Just stored more compactly.

## Summary

| Stage | What's quantized | Precision | Why |
|---|---|---|---|
| Inference weights | W matrices | INT4/INT8/FP8 | Reduce memory bandwidth |
| Inference activations | X tensors | Sometimes INT8 | Faster tensor core ops |
| Training forward | Activations | FP16/BF16 | Save activation memory |
| Training backward | Gradients | FP16/BF16 | Speed |
| Training update | Master weights | FP32 | Prevent gradient underflow |
| QLoRA base | Frozen weights | 4-bit | Fit large models in small GPU |
| QLoRA adapters | LoRA weights | FP16 | Need gradient precision |
| Distillation teacher | Frozen weights | FP8/INT8 OK | Only forward pass |
| Distillation student | Training weights | FP16 + FP32 master | Learning |
