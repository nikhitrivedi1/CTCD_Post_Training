# Config Parameters Explained

A comprehensive guide to every configuration parameter used in our knowledge distillation training.

---

## Table of Contents

1. [Training Parameters](#training-parameters)
2. [Distillation Parameters](#distillation-parameters)
3. [Model Parameters](#model-parameters)
4. [Data Parameters](#data-parameters)
5. [Checkpointing Parameters](#checkpointing-parameters)
6. [Evaluation Parameters](#evaluation-parameters)
7. [Logging Parameters](#logging-parameters)
8. [Quick Reference](#quick-reference)

---

## Training Parameters

### `batch_size`

Number of samples processed before one gradient update.

```
Step 1: Process 64 samples → compute average gradient → update weights
Step 2: Process next 64 samples → compute average gradient → update weights
...

Effects:
├── Larger batch → more stable gradients, but may overfit
├── Smaller batch → noisier gradients, better generalization
├── Memory usage scales linearly with batch size
└── Must scale learning_rate proportionally (linear scaling rule)
```

**Typical values:** 16 (paper), 64-256 (practical), 512+ (large GPU)

---

### `learning_rate`

The step size for weight updates.

```
new_weight = old_weight - learning_rate × gradient

Too high (0.01):   Overshoots minimum, diverges, NaN loss
Too low (1e-7):    Converges extremely slowly
Just right (2e-5): Stable convergence to good minimum
```

**Scaling rule:** When batch size increases by k×, multiply learning rate by k×.

```
Batch 16  → lr = 2e-5    (base)
Batch 64  → lr = 8e-5    (4× base)
Batch 256 → lr = 3.2e-4  (16× base)
Batch 512 → lr = 6.4e-4  (32× base, cap at ~1e-3)
```

**Typical values:** 2e-5 (base), scaled up to 1e-3 max for fine-tuning

---

### `warmup_steps`

Gradually increases learning rate from 0 to target over N steps.

```
Why warmup matters:
─────────────────────

At training start (step 0):
├── Adam's moment estimates (m, v) are initialized to 0
├── Gradient estimates are unreliable (random weights)
├── Large learning rate → wild, unstable updates
└── Can cause irreversible damage to model

With warmup (e.g., 300 steps):
Step    0: lr = 0.0
Step  100: lr = 0.000027  (1/3 of target)
Step  200: lr = 0.000053  (2/3 of target)
Step  300: lr = 0.00008   (full target)  ← warmup complete
Step  301+: lr starts decaying toward 0

Visual:
    lr
     │       ╱╲
     │      ╱  ╲
     │     ╱    ╲
     │    ╱      ╲____
     │   ╱             ╲____
     │  ╱                   ╲____
     │ ╱
     └─┴─────────────────────────→ steps
       ↑
    warmup        decay phase
     ends
```

**Rule of thumb:** 5-10% of total training steps, or scale with batch size.

```
Batch 16:  warmup = 200  (paper setting)
Batch 64:  warmup = 200-300
Batch 256: warmup = 500
Batch 512: warmup = 500-800
Batch 1024: warmup = 1000+
```

---

### `num_epochs`

Number of complete passes through the training dataset.

```
MNLI dataset has ~393,000 training samples

Epoch 1: See all 393K samples (in random order)
Epoch 2: See all 393K samples again (reshuffled)
Epoch 3: See all 393K samples again (reshuffled)

Total samples seen: 393K × 3 = ~1.18 million
```

**Typical values:** 3 (standard for fine-tuning), 5+ (if underfitting)

---

### `weight_decay`

L2 regularization applied to weights to prevent overfitting.

```
total_loss = task_loss + weight_decay × Σ(weights²)

Effect:
├── Penalizes large weight values
├── Encourages simpler, more generalizable models
├── Prevents overfitting to training data
└── AdamW applies this separately from gradient (decoupled)

Visual intuition:
Without weight decay: Weights can grow arbitrarily large
With weight decay:    Weights "decay" toward zero unless needed
```

**Typical values:** 0.01 (standard for transformers), 0.0 (no regularization)

---

### `max_grad_norm`

Gradient clipping - caps gradient magnitude to prevent exploding gradients.

```
If ||gradient|| > max_grad_norm:
    gradient = gradient × (max_grad_norm / ||gradient||)

Example:
├── Gradient magnitude: 15.0
├── max_grad_norm: 2.0
├── Clipped gradient: gradient × (2.0 / 15.0) = gradient × 0.133
└── New magnitude: 2.0

Prevents:
├── NaN losses from exploding gradients
├── Training instability
├── Wild updates from outlier batches
└── Divergence during early training
```

**Typical values:** 1.0-2.0 (transformers), 5.0+ (more lenient)

---

### `fp16` / `bf16`

Mixed precision training using 16-bit floating point numbers.

```
fp16: true  → Use FP16 (half precision)
bf16: true  → Use BF16 (brain float, better range)
Both false  → Use FP32 (full precision)

Benefits of mixed precision:
├── 2× less memory for activations
├── 2× faster on Tensor Cores
├── Can fit larger batch sizes
└── Usually no accuracy loss

FP16 vs BF16:
├── FP16: Better precision, risk of overflow (needs loss scaling)
├── BF16: Same range as FP32, less precision, more stable
└── Use BF16 on A100/H100, FP16 on V100/T4

When to disable:
├── MPS (Apple Silicon) - doesn't support well
├── Debugging NaN issues
└── When exact reproducibility needed
```

---

### `dataloader_num_workers`

Number of CPU processes for parallel data loading.

```
num_workers=0: Main process loads data (slowest, but simplest)
num_workers=4: 4 parallel workers prepare batches
num_workers=8: 8 parallel workers prepare batches

How it works:
┌─────────────────────────────────────────┐
│  Worker 1 → [batch 1] [batch 5] [batch 9]
│  Worker 2 → [batch 2] [batch 6] [batch 10]
│  Worker 3 → [batch 3] [batch 7] [batch 11]
│  Worker 4 → [batch 4] [batch 8] [batch 12]
│                    ↓
│            Prefetch Queue
│                    ↓
│                  GPU
└─────────────────────────────────────────┘

More workers = GPU never waits for data
But: Uses more CPU memory, diminishing returns past 8-16
```

**Typical values:** 0 (MPS), 2 (Colab), 4-8 (local), 8-16 (HPC)

---

### `dataloader_pin_memory`

Pre-allocate memory in CPU RAM for faster GPU transfer.

```
pin_memory: false
├── Data loaded to pageable memory
├── Must copy to pinned memory before GPU transfer
└── Extra copy operation

pin_memory: true
├── Data loaded directly to pinned (page-locked) memory
├── DMA transfer directly to GPU
└── Faster CPU→GPU transfer

Always use True on CUDA GPUs.
Use False on MPS (Apple) or CPU-only.
```

---

## Distillation Parameters

### `temperature`

Softens probability distributions for knowledge transfer.

```
Formula: softmax(logits / temperature)

Temperature = 1 (no softening):
├── Teacher output: [0.95, 0.03, 0.02]
├── Very confident, almost one-hot
└── Student only learns "correct answer is class 0"

Temperature = 2 (moderate softening):
├── Teacher output: [0.70, 0.18, 0.12]
├── Reveals relative similarities between classes
└── Student learns "class 0, but class 1 is more similar than class 2"

Temperature = 5 (heavy softening):
├── Teacher output: [0.45, 0.30, 0.25]
├── Almost uniform, maximum "dark knowledge"
└── May lose too much signal

Visual:
T=1:  ████████████░░  (sharp, confident)
T=2:  ████████░░░░░░  (softer, informative)
T=5:  █████░░░░░░░░░  (very soft, may be too uniform)
```

**Typical values:** 2-4 (general), 11 (QQP in paper), 1 (no softening)

---

### `alpha`

Balance between distillation loss (KL) and task loss (CE).

```
total_loss = α × KL_divergence(student, teacher)
           + (1-α) × CrossEntropy(student, labels)

α = 0.0: Only hard labels
├── No knowledge from teacher
├── Standard fine-tuning baseline
└── Use for ablation study

α = 0.5: Equal weight (paper default)
├── Balanced learning from both sources
├── Usually best performance
└── Standard distillation setting

α = 1.0: Only soft labels
├── Pure distillation, no ground truth
├── Tests if teacher knowledge is sufficient
└── Use for ablation study

Visual:
α=0.0: [████████████] CE only
α=0.5: [██████][██████] KL + CE
α=1.0: [████████████] KL only
```

**Typical values:** 0.5 (balanced), 0.0/1.0 (ablation)

---

## Model Parameters

### `teacher`

The large, pre-trained model providing "soft labels".

```
teacher: bert-base-uncased

├── 110M parameters
├── 12 transformer layers
├── 768 hidden dimension
├── Pre-trained on large corpus
├── Frozen during distillation (no gradient updates)
└── Provides soft probability distributions for student
```

### `student`

The smaller model being trained to mimic the teacher.

```
student: distilbert-base-uncased

├── 66M parameters (40% smaller)
├── 6 transformer layers (50% fewer)
├── 768 hidden dimension (same)
├── Learns from teacher + labels
└── Goal: Match teacher accuracy with fewer parameters
```

### `student_init`

How to initialize student weights before training.

```
student_init: fresh
├── Random initialization
├── Student learns everything from scratch
└── Fair comparison, but slower convergence

student_init: pretrained
├── Use DistilBERT's pretrained weights
├── Already has language understanding
└── Faster convergence, may be "unfair" comparison
```

---

## Data Parameters

### `task`

The dataset/task to train on.

```
task: mnli
├── Multi-Genre Natural Language Inference
├── 393K training samples
├── 3 classes: entailment, neutral, contradiction
├── Input: premise + hypothesis sentence pairs
└── Target accuracy: 81.7% (paper)

task: qqp
├── Quora Question Pairs
├── 364K training samples
├── 2 classes: duplicate, not duplicate
├── Input: question + question pairs
└── Target accuracy: 90.0% (paper)
```

### `max_length`

Maximum sequence length in tokens after tokenization.

```
max_length: 128

├── Longer → More context, more memory, slower
├── Shorter → Less context, less memory, faster
├── Sequences longer than max_length are truncated
├── Sequences shorter are padded
└── 128 is standard for sentence-pair classification

Memory scales as O(max_length²) due to attention!

max_length: 128 → 1× memory
max_length: 256 → 4× memory
max_length: 512 → 16× memory
```

---

## Checkpointing Parameters

### `save_steps`

Save a model checkpoint every N training steps.

```
save_steps: 500

With batch=64 on MNLI (~393K samples):
├── Steps per epoch: ~6,136
├── Checkpoints per epoch: ~12
├── Saved at: step 500, 1000, 1500, 2000, ...

Each checkpoint contains:
├── Model weights
├── Optimizer state
├── Training step number
├── Random number generator states
└── Allows exact resume from any checkpoint
```

### `save_total_limit`

Maximum number of checkpoints to keep on disk.

```
save_total_limit: 3

├── Keeps only the 3 most recent checkpoints
├── Automatically deletes older ones
├── Prevents disk from filling up
└── Always keeps the "best" checkpoint if load_best_model_at_end=True

Example with save_steps=500, save_total_limit=3:
Step 500:  checkpoint-500 (kept)
Step 1000: checkpoint-500, checkpoint-1000 (kept)
Step 1500: checkpoint-500, checkpoint-1000, checkpoint-1500 (kept)
Step 2000: checkpoint-1000, checkpoint-1500, checkpoint-2000 (500 deleted)
```

---

## Evaluation Parameters

### `eval_steps`

Run evaluation on validation set every N training steps.

```
eval_steps: 500

IMPORTANT: Must match save_steps for load_best_model_at_end to work!

Every 500 steps:
├── Pause training
├── Run model on validation set (~10K samples)
├── Compute metrics (loss, accuracy)
├── Log to W&B
├── Save checkpoint if best so far
└── Resume training

Trade-off:
├── Frequent eval → Better tracking, slower training
└── Rare eval → Faster training, may miss best checkpoint
```

### `eval_strategy`

When to run evaluation.

```
eval_strategy: "steps"
├── Evaluate every eval_steps steps
├── Most control over frequency
└── Recommended for long training runs

eval_strategy: "epoch"
├── Evaluate at end of each epoch
├── Simple, but may be too infrequent
└── Good for short training runs

eval_strategy: "no"
├── Never evaluate during training
├── Only evaluate at the end
└── Fastest, but no early stopping possible
```

---

### Checkpoint Selection (Best Model)

The trainer selects the "best" checkpoint based on the task's primary metric, not loss.

```
Classification tasks:
├── MNLI, SST-2 → best eval_accuracy
├── QQP → best eval_f1
├── CoLA → best eval_mcc
└── Translation → best eval_bleu (or eval_loss if no metric)

Why metric-based selection?
├── Loss and metric can diverge during training
├── A checkpoint may have lower loss but worse accuracy
├── Selecting by metric ensures we report the best achieved
└── This is standard practice for ML evaluation

Example divergence:
Step 6000: accuracy=0.81, loss=0.45  ← best accuracy
Step 8000: accuracy=0.80, loss=0.42  ← best loss
→ Metric-based selects step 6000 (0.81 accuracy)
→ Loss-based would select step 8000 (0.80 accuracy)

The selected metric is logged to W&B config as `metric_for_best_model`.
```

---

## Logging Parameters

### `report_to`

Where to send training logs and metrics.

```
report_to: "wandb"
├── Weights & Biases cloud dashboard
├── Real-time loss curves, GPU usage, etc.
├── Requires WANDB_API_KEY
└── Best for tracking experiments

report_to: "tensorboard"
├── Local TensorBoard logging
├── Run: tensorboard --logdir=./outputs
└── No internet required

report_to: "none"
├── No logging to external services
├── Only console output
└── Fastest, but no visualization
```

### `logging_steps`

Log training metrics every N steps.

```
logging_steps: 100

Every 100 steps, log:
├── Training loss
├── Learning rate (current)
├── Gradient norm
├── Throughput (samples/sec)
├── GPU memory usage
└── Estimated time remaining

Trade-off:
├── Frequent logging → Better visibility, slight overhead
└── Rare logging → Faster, less insight
```

---

## Quick Reference

```
┌─────────────────────────────────────────────────────────────────┐
│                 CONFIG PARAMETER CHEAT SHEET                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TRAINING DYNAMICS (affect model quality):                       │
│  ─────────────────────────────────────────                       │
│  batch_size      → Gradient quality vs memory                   │
│  learning_rate   → Step size (MUST scale with batch!)           │
│  warmup_steps    → Stabilize early training                     │
│  weight_decay    → L2 regularization (prevent overfit)          │
│  max_grad_norm   → Clip gradients (prevent explosion)           │
│  num_epochs      → How many passes through data                 │
│                                                                  │
│  DISTILLATION (knowledge transfer):                              │
│  ─────────────────────────────────                               │
│  temperature     → Soften teacher outputs (2-4 typical)         │
│  alpha           → Balance KL vs CE loss (0.5 typical)          │
│                                                                  │
│  PERFORMANCE (speed/memory):                                     │
│  ───────────────────────────                                     │
│  fp16/bf16       → Mixed precision (2× faster, 2× less mem)     │
│  num_workers     → Parallel data loading (4-8 typical)          │
│  pin_memory      → Faster CPU→GPU transfer (always True)        │
│  max_length      → Sequence length (memory ∝ length²)           │
│                                                                  │
│  CHECKPOINTING:                                                  │
│  ──────────────                                                  │
│  save_steps      → Checkpoint frequency                         │
│  eval_steps      → Validation frequency (MUST match save!)      │
│  save_total_limit→ Max checkpoints on disk                      │
│                                                                  │
│  LOGGING:                                                        │
│  ────────                                                        │
│  report_to       → wandb, tensorboard, or none                  │
│  logging_steps   → How often to log metrics                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Common Configurations

### Paper Reproduction (Exact Match)
```yaml
batch_size: 16
learning_rate: 0.00002
warmup_steps: 200
temperature: 2
alpha: 0.5
```

### Fast Iteration (Development)
```yaml
batch_size: 256
learning_rate: 0.00032  # 16× scaled
warmup_steps: 500
temperature: 2
alpha: 0.5
```

### Maximum Speed (Hyperparameter Search)
```yaml
batch_size: 512
learning_rate: 0.00064  # 32× scaled
warmup_steps: 500
num_epochs: 2           # Fewer epochs OK for comparison
```

### Ablation: No Distillation
```yaml
alpha: 0.0              # Only CE loss
# Everything else same as paper
```

### Ablation: Pure Distillation
```yaml
alpha: 1.0              # Only KL loss
# Everything else same as paper
```
