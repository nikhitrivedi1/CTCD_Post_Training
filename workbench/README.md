# Workbench

Post-hoc analysis tools and experiments for understanding distillation results.

## Usage

```bash
# Activate environment
source ~/.zshrc && conda activate distillation

# Linear probing (teacher only)
python workbench/analyze.py --teacher bert-base-uncased --task mnli --test probe

# Linear probing (compare teacher and student)
python workbench/analyze.py \
    --teacher bert-base-uncased \
    --student outputs/mnli_cotrain_sft/checkpoint \
    --task mnli \
    --test probe \
    --output workbench/results/probing_results.json

# CKA similarity analysis
python workbench/analyze.py \
    --teacher bert-base-uncased \
    --student outputs/mnli_cotrain_sft/checkpoint \
    --task mnli \
    --test cka \
    --output workbench/results/cka_results.json

# Run all analyses
python workbench/analyze.py \
    --teacher bert-base-uncased \
    --student outputs/mnli_cotrain_sft/checkpoint \
    --task mnli \
    --test all
```

## Available Commands

### `probe` - Linear Probing

Trains linear classifiers on frozen hidden states at each layer to measure where task-relevant information emerges.

**What it tells you:**
- At which layer does task signal saturate?
- Does student reach teacher's peak accuracy?
- Are later student layers redundant?

**Arguments:**
| Arg | Description | Default |
|-----|-------------|---------|
| `--teacher` | Teacher model (required) | - |
| `--student` | Student checkpoint (optional) | None |
| `--task` | Task (mnli, qqp, sst2, cola) | mnli |
| `--output` | Output JSON file | None |
| `--max_samples` | Samples for probing | Full dataset |

**Example output:**
```
Teacher (bert-base-uncased):
  embeddings: 45.2%
  layer_1: 52.1%
  ...
  layer_12: 82.4%

Student (checkpoint):
  embeddings: 48.3%
  ...
  layer_6: 81.2%

Comparison:
  Teacher peak: layer_12 @ 82.4%
  Student peak: layer_6 @ 81.2%
  Retention: 98.5%
```

### `cka` - CKA Similarity

Computes Centered Kernel Alignment between student and teacher layer representations.

**What it tells you:**
- Which student layers align with which teacher layers?
- Does the student develop similar internal structure?
- Are there layer mapping mismatches?

**Arguments:**
| Arg | Description | Default |
|-----|-------------|---------|
| `--teacher` | Teacher model (required) | - |
| `--student` | Student checkpoint (required) | - |
| `--task` | Task | mnli |
| `--output` | Output JSON file | None |
| `--max_samples` | Samples (CKA is O(N^2)) | 1000 |

**Example output:**
```
CKA ALIGNMENT

Best teacher layer match for each student layer:
  embeddings -> embeddings (CKA: 0.95)
  layer_1 -> layer_2 (CKA: 0.87)
  layer_2 -> layer_4 (CKA: 0.82)
  layer_3 -> layer_6 (CKA: 0.79)
  ...
```

## Interpreting Results

### Probing Curves

| Pattern | Interpretation |
|---------|----------------|
| Student saturates early | Later layers may be redundant - try fewer layers |
| Student never reaches teacher | Capacity gap - try wider/deeper student |
| Student peaks higher than teacher | Student learned task-specific features teacher didn't |

### CKA Alignment

| Pattern | Interpretation |
|---------|----------------|
| Diagonal alignment | Student layers match expected teacher layers |
| Off-diagonal peaks | Student layer N learns teacher layer M features |
| Low overall CKA | Student developing different representations |

## Module Structure

```
workbench/
├── __init__.py        # Exports
├── analyze.py         # CLI entry point
├── probing.py         # Linear probing
├── cka.py             # CKA similarity
├── attention.py       # Attention entropy analysis
├── disagreement.py    # Prediction disagreement analysis
├── finetune_head.py   # Head fine-tuning with frozen encoder
├── utils.py           # Shared utilities
├── results/           # Output directory
└── README.md          # This file
```

## Adding New Analyses

1. Create `workbench/new_analysis.py`
2. Implement `run_new_analysis()` function
3. Add command to `analyze.py`
