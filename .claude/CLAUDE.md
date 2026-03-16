# Claude Rules for Forge

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
- **Defensive programming**: Make the safe behavior the default. Don't rely on each code path to remember invariants - enforce them in one place (e.g., `.to(device)` at the end of every loader, not scattered per-branch).

## Git Workflow
- Commit messages: short summary line, optional body for context.
- Do NOT mention LLM assistance anywhere (commits, PRs, code comments, docs) unless explicitly requested. This includes Co-Authored-By, "Generated with", "AI-assisted", or similar attributions.
- Push when explicitly asked.
- **DCP**: When the user says "DCP", do: **D**ouble-check the code, **C**ommit, **P**ush.
- **Merging**: Always squash-and-merge branches into main.

## Project Context
- This is an open training framework (FT + KD + RL).
- Main components: forge.py (entry point), src/training/trainer.py (DistillationTrainer).
- Configs in configs/ use YAML with inheritance (inherit: base.yaml).
- FAR factorial study: 8 combinations of Feature/Attention/Response distillation signals.
- Runs output to outputs/{run_name}/ with config.yaml, run_metadata.json, output.log.

## Python Environment
- Always activate conda environment before running Python commands:
  ```bash
  source ~/.zshrc && conda activate forge
  ```
- Environment: forge (Python 3.11)
- Dependencies in environment.yml
- **Never install packages directly** via `pip install` or `conda install`. Always:
  1. Add the dependency to `environment.yml`
  2. Run `conda env update --file environment.yml` to install
- For optional heavyweight deps (e.g. playwright + browser binaries), make them gracefully optional in code

## Dates
- The current year is 2026. Use this when writing dates in docs, logs, etc.

## Cluster Workflow
- Cluster: explorer (login.explorer.northeastern.edu), work dir: /scratch/ravi.ka/Forge
- **Always use tmux sessions for cluster commands, never one-shot `ssh` commands.**
  - Create session: `tmux new-session -d -s cluster 'ssh -t explorer "cd /scratch/ravi.ka/Forge && bash"'`
  - Run commands: `tmux send-keys -t cluster 'command' Enter`
  - Read output: `sleep N && tmux capture-pane -t cluster -p | tail -M`
  - Only exception: simple, fast, non-interactive commands like `git pull` can use direct ssh.
- Use `python` not `python3` on the cluster

## Testing
- **TDD**: When fixing bugs or changing behavior, write a failing test first, then implement the fix.
- GPU clusters: H200 (140GB), A100 (40GB).
- Watch for OOM during evaluation - hidden states/attentions accumulate.
- Use --debug flag for verbose logging and W&B Test project.
