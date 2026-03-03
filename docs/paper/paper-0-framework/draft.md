# Paper 0: A Unified Framework for Knowledge Distillation Research

**Status:** Draft
**Target:** Framework/Systems paper or Appendix to main papers

---

## Abstract

We present a unified, config-driven framework for systematic knowledge distillation research. The framework supports both classification (BERT → DistilBERT) and generation (Llama 3B → 1B) tasks with a consistent experimental interface. Key contributions include: (1) YAML-based config inheritance for reproducible experiments, (2) FAR factorial design for ablating distillation signals, (3) automatic CE-only baseline detection, and (4) integrated W&B logging with structured tagging.

---

## 1. Introduction

Knowledge distillation research suffers from reproducibility issues:
- Inconsistent experimental setups across papers
- Ad-hoc implementations that don't generalize
- Difficulty comparing results across different codebases

We address these with a unified framework that:
- **Separates concerns**: Config (what) vs Code (how)
- **Enables factorial experiments**: Systematic ablation of distillation signals
- **Supports multiple modalities**: Classification and generation tasks
- **Provides reproducibility**: Deterministic runs with seed control

---

## 2. Framework Architecture

### 2.1 Config-Driven Design

```yaml
# Inheritance enables experiment variations
inherit: base.yaml

model:
  teacher: bert-base-uncased
  student: distilbert-base-uncased

distillation:
  alpha: 0.5           # CE vs KL balance
  temperature: 2.0     # Softmax temperature
  freeze_teacher: true # Frozen vs co-training
  loss_type: kl        # kl, mse, cosine
```

**Benefits:**
- Experiment configs are self-documenting
- Easy to create variations via inheritance
- Git-trackable experimental history

### 2.2 Modular Loss Strategies

```
LossStrategy (abstract)
├── KLDivergenceLoss      # Response distillation (R)
├── FeatureMatchingLoss   # Hidden state matching (F)
├── AttentionMatchingLoss # Attention transfer (A)
└── CombinedLoss          # Weighted combination
```

Each strategy computes:
- Forward pass requirements (hidden states, attentions)
- Loss computation
- Logging metrics

### 2.3 Teacher-Student Loading

```python
# Automatic CE-only detection
if teacher_name is None or alpha == 0.0:
    return _load_student_only(...)  # Skip teacher
```

**Key features:**
- Memory-efficient: No teacher loaded when not needed
- LoRA support for LLM distillation
- Automatic embedding resizing for custom tokens

---

## 3. FAR Factorial Design

### 3.1 Distillation Signals

| Signal | Symbol | Loss | What Transfers |
|--------|--------|------|----------------|
| Response | R | KL divergence | Output distribution |
| Feature | F | MSE on hidden | Internal representations |
| Attention | A | MSE on attention | Attention patterns |

### 3.2 Factorial Combinations

```
00: CE-only (baseline)
01: F-only
02: A-only
03: R-only (classic KD)
04: FA
05: FR
06: AR
07: FAR (all signals)
```

**Design rationale:**
- 2³ = 8 combinations for complete coverage
- CE-only baseline for significance testing
- Individual signals identify most valuable transfer

### 3.3 Config Organization

```
configs/
├── classification/
│   └── mnli/bert_distilbert/
│       ├── base.yaml           # Shared settings
│       ├── far/                # Factorial study
│       │   ├── 00_ce-only.yaml
│       │   ├── 01_f-only.yaml
│       │   └── ...
│       └── ablations/          # Frozen vs co-train
└── translation/
    └── europarl/llama3p2-3b_llama3p2-1b/
        ├── base.yaml
        └── far/
```

---

## 4. Experiment Management

### 4.1 Run Tracking

Each run produces:
```
outputs/{run_name}/
├── config.yaml        # Resolved config (no inheritance)
├── run_metadata.json  # Git SHA, timestamps, GPU info
├── output.log         # Training logs
└── checkpoints/       # Model checkpoints
```

### 4.2 W&B Integration

**Tagging convention:**
```yaml
wandb_tags: ["mnli", "far", "r-only", "frozen"]
```

**Structured logging:**
- Config parameters → W&B config
- Metrics → W&B summary
- Loss curves → W&B history

### 4.3 Reproducibility

```bash
# Reproduce any run
python forge.py --config outputs/run_name/config.yaml --seed 42
```

---

## 5. Supported Tasks

### 5.1 Classification (GLUE)

| Task | Metric | Teacher | Student |
|------|--------|---------|---------|
| MNLI | Accuracy | BERT-base | DistilBERT |
| QQP | F1 | BERT-base | DistilBERT |
| SST-2 | Accuracy | BERT-base | DistilBERT |
| CoLA | MCC | BERT-base | DistilBERT |

### 5.2 Translation (Europarl)

| Task | Metric | Teacher | Student |
|------|--------|---------|---------|
| EN→FR | BLEU, COMET | Llama-3.2-3B | Llama-3.2-1B |

**Translation-specific features:**
- Chat template formatting
- Label masking (only target tokens contribute to loss)
- LoRA for parameter-efficient fine-tuning
- BLEU/COMET evaluation via generation

---

## 6. Implementation Details

### 6.1 Training Loop

```python
# DistillationTrainer extends HuggingFace Trainer
class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs):
        # Student forward pass
        student_outputs = model(**inputs)

        # Teacher forward pass (if needed)
        if self.teacher is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher(**inputs)

        # Compute combined loss
        return self.loss_strategy(
            student_outputs, teacher_outputs, inputs["labels"]
        )
```

### 6.2 Memory Optimization

- **Gradient checkpointing**: For large models
- **Mixed precision (fp16)**: Automatic on CUDA
- **Teacher freezing**: No gradients stored for teacher
- **LoRA**: Only ~0.6% parameters trainable for LLMs

---

## 7. Usage Examples

### Basic Training

```bash
python forge.py --config configs/classification/mnli/bert_distilbert/far/03_r-only.yaml
```

### Debug Mode

```bash
python forge.py --config ... --debug --max_steps 10
```

### Batch Experiments

```bash
# Run all FAR combinations
for config in configs/classification/mnli/bert_distilbert/far/*.yaml; do
    python forge.py --config $config --timestamp
done
```

---

## 8. Extensibility

### Adding a New Task

1. Add to `src/tasks/__init__.py`
2. Create config hierarchy in `configs/`
3. Implement data loading if needed

### Adding a New Loss Strategy

1. Subclass `LossStrategy` in `src/training/losses/`
2. Register in strategy factory
3. Add `loss_type` option to configs

### Adding a New Model Pair

1. Update `src/models/loader.py` if needed
2. Create config with new teacher/student names
3. Verify tokenizer compatibility

---

## 9. Conclusion

This framework enables:
- **Reproducible** distillation experiments
- **Systematic** ablation via factorial design
- **Generalizable** findings across tasks/modalities

All code is available at: [repository URL]

---

## Appendix: Config Reference

See [config-parameters.md](../../guides/config-parameters.md) for full parameter documentation.
