# Claude Rules for Distillation Project

## Communication Style
- Be concise. Short answers are preferred over verbose explanations.
- Don't ask unnecessary clarifying questions - make reasonable decisions.
- When the user provides feedback or a preference, apply it without lengthy justification.
- "clear" means acknowledged, move on.

## Code Style
- Include docstrings for functions with Args/Returns sections.
- Use descriptive variable names.
- Prefer general naming over overly specific (e.g., `output.log` not `training.log`).
- Add comments only where logic isn't self-evident.

## Git Workflow
- Commit messages: short summary line, optional body for context.
- Do NOT include Co-Authored-By unless explicitly requested.
- Push when explicitly asked.

## Project Context
- This is a knowledge distillation research project for CS-7150 Deep Learning.
- Main components: distill.py (entry point), src/training/trainer.py (DistillationTrainer).
- Configs in configs/ use YAML with inheritance (inherit: base.yaml).
- FAR factorial study: 8 combinations of Feature/Attention/Response distillation signals.
- Runs output to outputs/{run_name}/ with config.yaml, run_metadata.json, output.log.

## Python Environment
- Always activate conda environment before running Python commands:
  ```bash
  source ~/.zshrc && conda activate distillation
  ```
- Environment: distillation (Python 3.10)
- Dependencies in environment.yml

## Dates
- The current year is 2026. Use this when writing dates in docs, logs, etc.

## Testing
- GPU clusters: H200 (140GB), A100 (40GB).
- Watch for OOM during evaluation - hidden states/attentions accumulate.
- Use --debug flag for verbose logging and W&B Test project.
