# Analysis

W&B data analysis workflow for experiment results.

---

## Structure

```
analysis/
├── data/           # Downloaded W&B data (CSV/JSON)
├── scripts/        # Python scripts for fetching & processing
├── notebooks/      # Jupyter analysis notebooks
└── summaries/      # Markdown summaries of findings
```

---

## Workflow

### 1. Fetch Data from W&B

```bash
python docs/analysis/scripts/fetch_wandb.py --output docs/analysis/data/
```

### 2. Analyze in Notebooks

Open notebooks in `notebooks/` for interactive analysis:
- `ablation_analysis.ipynb` - MNLI/QQP ablation study
- `far_analysis.ipynb` - FAR factorial study
- `translation_analysis.ipynb` - Europarl translation

### 3. Generate Summaries

Export key findings to `summaries/` as markdown for paper integration.

---

## Data Files

| File | Description | Last Updated |
|------|-------------|--------------|
| `runs_YYYY-MM-DD.csv` | All runs snapshot | - |
| `mnli_ablation.csv` | MNLI ablation metrics | - |
| `mnli_far.csv` | MNLI FAR study metrics | - |

---

## Key Metrics

### Classification (GLUE)
- `eval_accuracy` - Primary metric for MNLI, SST-2
- `eval_f1` - Primary metric for QQP
- `eval_mcc` - Primary metric for CoLA
- `eval_loss` - Cross-entropy loss

### Translation (Europarl)
- `bleu` - BLEU score (sacrebleu)
- `comet` - COMET score (neural metric)
- `eval_loss` - Cross-entropy loss

---

## W&B Projects

| Project | Entity | Description |
|---------|--------|-------------|
| Forge-Default | karthik-ravi-1537-personal | Main experiments |
| Test | karthik-ravi-1537-personal | Debug/test runs |
