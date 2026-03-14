# Chain Orchestrator: tmux Guide

The chain orchestrator runs on a login node inside tmux so it survives SSH disconnects.

## Quick Reference

```bash
# Create session and start chain
tmux new -s chain
scripts/slurm/run_chain.sh chains/qwen35-27b_ocr_pipeline.yaml

# Detach: Ctrl+B, then D

# Reattach (must be on the same login node)
tmux attach -t chain

# Resume a partially-complete chain
tmux new -s chain
scripts/slurm/run_chain.sh --resume qwen35-27b_ocr_pipeline
```

## Login Node Pinning

tmux sessions are local to each login node (explorer-01, explorer-02, etc.). If you SSH in and land on a different node, your session won't be there.

Pin your SSH to a specific node:

```bash
ssh ravi.ka@explorer-01.explorer.northeastern.edu
```

Or check which node has your session:

```bash
# From any login node
ssh explorer-01 'tmux list-sessions' 2>/dev/null
ssh explorer-02 'tmux list-sessions' 2>/dev/null
```

## Monitoring

```bash
# Inside tmux: the orchestrator prints polling status every 60s
# Outside tmux: peek at the latest output
tmux capture-pane -t chain -p | tail -20

# Check chain state
cat .incomplete_chains

# Check what's running
squeue -u $USER
```

## Recovery

If the login node reboots or tmux dies, the chain state is saved in `.incomplete_chains`. Resume it:

```bash
tmux new -s chain
scripts/slurm/run_chain.sh --resume qwen35-27b_ocr_pipeline
```

The orchestrator reconciles: completed steps are skipped, active jobs are picked back up, failed steps are retried.

## Alternative: sbatch Mode

For long chains where login node stability is a concern, submit the orchestrator itself as a SLURM job:

```bash
sbatch scripts/slurm/run_chain.sh chains/qwen35-27b_ocr_pipeline.yaml
```

This runs on the `short` partition (48h max) and auto-resubmits via SIGUSR1. Monitor via `tail -f logs/chain_<jobid>.out`.
