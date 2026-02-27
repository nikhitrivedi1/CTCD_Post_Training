# Explorer HPC Guide

Complete guide for Northeastern's Explorer cluster, from login to debugging.

---

## Table of Contents

1. [Login & Setup](#login--setup)
2. [Environment Setup](#environment-setup)
3. [Understanding SLURM](#understanding-slurm)
4. [Job Submission (sbatch)](#job-submission-sbatch)
5. [Interactive Jobs (srun)](#interactive-jobs-srun)
6. [Job Monitoring](#job-monitoring)
7. [GPU Resources](#gpu-resources)
8. [Quality of Service (QOS)](#quality-of-service-qos)
9. [SLURM Environment Variables](#slurm-environment-variables)
10. [Debugging](#debugging)
11. [Data Transfer](#data-transfer)
12. [Best Practices](#best-practices)

---

## Login & Setup

### SSH Login

```bash
ssh <username>@login.explorer.northeastern.edu
```

**Common SSH options:**

```bash
# -X: Enable X11 forwarding for GUI applications
ssh -X <username>@login.explorer.northeastern.edu

# -L: Local port forwarding
ssh -L 8888:localhost:8888 <username>@login.explorer.northeastern.edu

# -A: Forward SSH agent (allows git push/pull without re-entering credentials)
ssh -A <username>@login.explorer.northeastern.edu
```

### SSH Config (recommended)

Add to `~/.ssh/config` on your **local machine**:

```
Host explorer
    HostName login.explorer.northeastern.edu
    User <username>
    ForwardAgent yes
    ServerAliveInterval 60
    ServerAliveCountMax 3

# Usage: ssh explorer
```

### First-Time Setup

```bash
# Navigate to scratch (fast storage, not backed up)
cd /scratch/<username>

# Clone your repository
git clone <your-repo-url>
cd <repo-name>

# Load conda module (required before using conda)
module load anaconda3/2024.06

# Create conda environment from file
conda env create -f environment.yml

# Activate the environment
conda activate distillation
```

### Directory Structure

| Path | Purpose | Quota | Speed | Backed Up |
|------|---------|-------|-------|-----------|
| `/home/<user>` | Config files, small scripts | 50GB | Slow (NFS) | Yes |
| `/scratch/<user>` | Code, data, model outputs | 10TB | Fast (Lustre) | **No** |
| `/work/<group>` | Shared group data | Varies | Medium | Yes |
| `/tmp` | Temporary files (per-node) | Node RAM | Very Fast | No |

**Recommendation:** Always work in `/scratch` for ML projects.

---

## Environment Setup

### Module System

```bash
# List ALL available modules
module avail

# Search for specific module
module avail cuda
module spider pytorch

# Load modules
module load cuda/12.1
module load anaconda3/2024.06

# Show currently loaded modules
module list

# Unload ALL modules (clean slate)
module purge
```

### Conda Environment

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate distillation

# Update environment from modified yml file
conda env update -f environment.yml --prune

# Verify GPU is accessible
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Understanding SLURM

SLURM (Simple Linux Utility for Resource Management) is the job scheduler.

### Key Concepts

| Term | Definition |
|------|------------|
| **Job** | A script/command submitted to run on compute nodes |
| **Node** | A physical server with CPUs, RAM, and optionally GPUs |
| **Partition** | A group of nodes with similar properties |
| **QOS** | Quality of Service - defines limits and priorities |
| **GRES** | Generic Resource - used for GPUs |

### SLURM Commands Overview

| Command | Purpose |
|---------|---------|
| `sbatch` | Submit a batch job script |
| `srun` | Run a command interactively |
| `squeue` | View queued and running jobs |
| `scancel` | Cancel a job |
| `sinfo` | View cluster/partition status |
| `sacct` | View job accounting/history |
| `scontrol` | View/modify job and node details |

---

## Job Submission (sbatch)

### Complete Example Script

```bash
#!/bin/bash
#SBATCH --job-name=distill
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --signal=SIGUSR1@120

# Print job info
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# Load required modules
module load cuda/12.1
module load anaconda3/2024.06

# Activate conda environment
source activate distillation

# Parse command line arguments
CONFIG=${1:-configs/default.yaml}
RUN_NAME=${2:-slurm_run}

# Run training
python distill.py \
    --config "$CONFIG" \
    --output_dir "./outputs/$RUN_NAME"
```

### Submitting Jobs

```bash
# Basic submission
sbatch script.sh

# With arguments
sbatch script.sh configs/my_config.yaml my_experiment

# Override directives from command line
sbatch --partition=short --time=01:00:00 script.sh
```

---

## Interactive Jobs (srun)

```bash
# Basic interactive GPU session
srun --partition=gpu --gres=gpu:1 --time=01:00:00 --pty bash

# Full options for interactive work
srun \
    --partition=gpu \
    --gres=gpu:h200:1 \
    --time=04:00:00 \
    --mem=64G \
    --cpus-per-task=8 \
    --pty bash
```

**Once inside the interactive session:**
```bash
nvidia-smi                      # Check GPU
module load cuda/12.1 anaconda3/2024.06
source activate distillation
python distill.py --max_steps 100  # Test your code
exit                            # End session
```

---

## Job Monitoring

### squeue - View Job Queue

```bash
# Your jobs
squeue -u $USER

# Custom format
squeue -u $USER -o "%.10i %.20j %.8T %.10M %.6D %R"

# Watch queue (refresh every 5 seconds)
watch -n 5 'squeue -u $USER'
```

#### Job States

| State | Code | Description |
|-------|------|-------------|
| PENDING | PD | Waiting in queue |
| RUNNING | R | Currently executing |
| COMPLETED | CD | Finished successfully |
| FAILED | F | Exited with non-zero code |
| TIMEOUT | TO | Killed for exceeding time limit |
| CANCELLED | CA | Cancelled by user or admin |

### sinfo - View Cluster Status

```bash
# Basic partition summary
sinfo

# GPU nodes status
sinfo -p gpu -N -o "%N %G %t %C %m"
```

### sacct - Job History

```bash
# Your recent jobs
sacct -u $USER

# Specific job details
sacct -j 12345 --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

### scancel - Cancel Jobs

```bash
# Cancel specific job
scancel 12345

# Cancel all your jobs
scancel -u $USER

# Cancel all pending jobs
scancel -u $USER --state=PENDING
```

---

## GPU Resources

### Available GPUs on Explorer

| GPU Type | VRAM | Best For |
|----------|------|----------|
| H200 | 80GB HBM3 | Large models, fastest training |
| A100-80GB | 80GB HBM2e | Large batch training |
| A100-40GB | 40GB HBM2e | Standard DL training |
| V100 | 32GB HBM2 | Medium workloads |
| P100 | 16GB HBM2 | Small models, testing |

### GPU Request Syntax

```bash
--gres=gpu:1                  # Any GPU type
--gres=gpu:a100:1             # 1 A100
--gres=gpu:h200:1             # 1 H200
--gres=gpu:a100:4             # 4 A100s (multi-GPU)
```

### Check GPU Availability

```bash
sinfo -p gpu -o "%20N %10G %10T %10c %10m"
```

---

## Quality of Service (QOS)

### Common QOS on Explorer

| QOS | MaxJobsPerUser | Max Time | Notes |
|-----|----------------|----------|-------|
| `gpu` | 4 | 24 hours | Standard GPU jobs |
| `gpu-short` | 2 | 4 hours | Quick GPU jobs |
| `short` | 50 | 4 hours | CPU only |
| `long` | 25 | 7 days | Extended runs |
| `courses-gpu` | 2 | 8 hours | Course projects |

### View QOS Limits

```bash
sacctmgr show qos format=Name,MaxJobsPU,MaxSubmitPU,MaxWall,Priority
```

---

## SLURM Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SLURM_JOB_ID` | Unique job identifier | `12345` |
| `SLURM_JOB_NAME` | Job name | `distill` |
| `SLURM_NODELIST` | Nodes allocated | `d4053` |
| `CUDA_VISIBLE_DEVICES` | GPU indices | `0` or `0,1` |
| `SLURM_ARRAY_TASK_ID` | Array task index | `3` |

---

## Debugging

### View Job Logs

```bash
cat logs/12345.out           # stdout
cat logs/12345.err           # stderr
tail -f logs/12345.out       # Follow in real-time
grep -i "error" logs/12345.err
```

### Common Errors

**CUDA Out of Memory:**
```python
# Reduce batch size or use gradient accumulation
batch_size = batch_size // 2
--gradient_accumulation_steps 2
```

**Job Timeout:**
```bash
# Request more time or implement checkpointing
#SBATCH --time=12:00:00
#SBATCH --signal=SIGUSR1@120
```

### SSH into Running Job

```bash
squeue -u $USER              # Find node (e.g., d4053)
ssh d4053                    # SSH to that node
nvidia-smi                   # Check GPU
```

---

## Data Transfer

```bash
# Upload files
scp file.txt username@login.explorer.northeastern.edu:/scratch/username/

# Download files
scp username@login.explorer.northeastern.edu:/scratch/username/results.tar.gz .

# rsync (better for large transfers)
rsync -avz --progress folder/ username@login.explorer.northeastern.edu:/scratch/username/folder/
```

---

## Best Practices

### 1. Project Organization
```
/scratch/<user>/project/
├── configs/
├── scripts/
├── logs/                  # gitignore
├── outputs/               # gitignore
├── src/
├── environment.yml
└── README.md
```

### 2. Always Test First
```bash
srun --partition=gpu --gres=gpu:1 --time=00:30:00 --pty bash
python distill.py --config config.yaml --max_steps 100
# Then submit full job
sbatch scripts/run_training.sh
```

### 3. Resource Estimation

| Model Size | GPU Memory | Recommended GPU |
|------------|------------|-----------------|
| ~66M (DistilBERT) | ~2GB | P100/V100 |
| ~110M (BERT) | ~4GB | V100/A100 |
| ~1B | ~20GB | A100-40GB |
| ~3B | ~40GB | A100-80GB/H200 |

---

## Quick Reference Card

| Task | Command |
|------|---------|
| Submit job | `sbatch script.sh` |
| Interactive GPU | `srun --partition=gpu --gres=gpu:1 --time=01:00:00 --pty bash` |
| View your jobs | `squeue -u $USER` |
| Cancel job | `scancel JOBID` |
| View cluster | `sinfo -p gpu` |
| Job history | `sacct -u $USER` |
| View logs | `tail -f logs/JOBID.out` |
| Check GPU | `nvidia-smi` |

### Module Setup
```bash
module load cuda/12.1 anaconda3/2024.06 && source activate distillation
```
