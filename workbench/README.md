# Workbench

Post-hoc analysis tools for comparing models and understanding training results.

## Usage

```bash
# Activate environment
source ~/.zshrc && conda activate forge

# Linear probing (single model)
python workbench/analyze.py --model bert-base-uncased --task mnli --test probe

# Linear probing (compare model against reference)
python workbench/analyze.py \
    --model outputs/mnli_cotrain_sft/checkpoint \
    --reference bert-base-uncased \
    --task mnli \
    --test probe \
    --output workbench/results/probing_results.json

# CKA similarity analysis
python workbench/analyze.py \
    --model outputs/mnli_cotrain_sft/checkpoint \
    --reference bert-base-uncased \
    --task mnli \
    --test cka \
    --output workbench/results/cka_results.json

# Run all analyses
python workbench/analyze.py \
    --model outputs/mnli_cotrain_sft/checkpoint \
    --reference bert-base-uncased \
    --task mnli \
    --test all
```

## Available Commands

### `probe` - Linear Probing

Trains linear classifiers on frozen hidden states at each layer to measure where task-relevant information emerges.

**What it tells you:**
- At which layer does task signal saturate?
- Does model reach reference's peak accuracy?
- Are later model layers redundant?

**Arguments:**
| Arg | Description | Default |
|-----|-------------|---------|
| `--model` | Model to analyze (required) | - |
| `--reference` | Reference model for comparison (optional) | None |
| `--task` | Task (mnli, qqp, sst2, cola) | mnli |
| `--output` | Output JSON file | None |
| `--max_samples` | Samples for probing | Full dataset |

**Example output:**
```
Model (checkpoint):
  embeddings: 48.3%
  ...
  layer_6: 81.2%

Reference (bert-base-uncased):
  embeddings: 45.2%
  layer_1: 52.1%
  ...
  layer_12: 82.4%

Comparison:
  Reference peak: layer_12 @ 82.4%
  Model peak: layer_6 @ 81.2%
  Retention: 98.5%
```

### `cka` - CKA Similarity

Computes Centered Kernel Alignment between model and reference layer representations.

**What it tells you:**
- Which model layers align with which reference layers?
- Does the model develop similar internal structure?
- Are there layer mapping mismatches?

**Arguments:**
| Arg | Description | Default |
|-----|-------------|---------|
| `--model` | Model to analyze (required) | - |
| `--reference` | Reference model (required) | - |
| `--task` | Task | mnli |
| `--output` | Output JSON file | None |
| `--max_samples` | Samples (CKA is O(N^2)) | 1000 |

**Example output:**
```
CKA ALIGNMENT

Best reference layer match for each model layer:
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
| Model saturates early | Later layers may be redundant - try fewer layers |
| Model never reaches reference | Capacity gap - try wider/deeper model |
| Model peaks higher than reference | Model learned task-specific features reference didn't |

### CKA Alignment

| Pattern | Interpretation |
|---------|----------------|
| Diagonal alignment | Model layers match expected reference layers |
| Off-diagonal peaks | Model layer N learns reference layer M features |
| Low overall CKA | Model developing different representations |

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
