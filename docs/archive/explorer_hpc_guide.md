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

**Parameters explained:**
- `<username>` - Your Northeastern username (e.g., `ravi.ka`)
- `login.explorer.northeastern.edu` - The login node hostname

**Common SSH options:**

```bash
# -X: Enable X11 forwarding for GUI applications (matplotlib plots, etc.)
ssh -X <username>@login.explorer.northeastern.edu

# -L: Local port forwarding - maps remote port to local port
# Format: -L <local_port>:<remote_host>:<remote_port>
ssh -L 8888:localhost:8888 <username>@login.explorer.northeastern.edu
#     ↑     ↑         ↑
#     |     |         └── Remote port (where Jupyter runs on cluster)
#     |     └──────────── Remote host (localhost = the login node itself)
#     └────────────────── Local port (access via localhost:8888 on your machine)

# -A: Forward SSH agent (allows git push/pull without re-entering credentials)
ssh -A <username>@login.explorer.northeastern.edu

# -v: Verbose mode for debugging connection issues
ssh -v <username>@login.explorer.northeastern.edu

# -o: Pass SSH options inline
ssh -o "ServerAliveInterval=60" <username>@login.explorer.northeastern.edu
#       └── Send keepalive every 60 seconds to prevent timeout
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

**Config options explained:**
| Option | Description |
|--------|-------------|
| `Host explorer` | Alias name - use `ssh explorer` instead of full command |
| `HostName` | Actual server address |
| `User` | Your username (auto-fills so you don't type it) |
| `ForwardAgent yes` | Forward SSH keys for git operations |
| `ServerAliveInterval 60` | Send keepalive packet every 60 seconds |
| `ServerAliveCountMax 3` | Disconnect after 3 failed keepalives (3 min timeout) |

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

**Recommendation:** Always work in `/scratch` for ML projects. Files in `/scratch` are **not backed up** - push code to git regularly.

---

## Environment Setup

### Module System

The module system manages software versions. You **must** load modules before using software like CUDA or conda.

```bash
# List ALL available modules (long output)
module avail

# Search for specific module
module avail cuda              # Shows all CUDA versions
module avail anaconda          # Shows all Anaconda versions
module spider pytorch          # Deep search including dependencies

# Load a module (adds to PATH, sets environment variables)
module load cuda/12.1
module load anaconda3/2024.06

# Load multiple modules at once
module load cuda/12.1 anaconda3/2024.06

# Show currently loaded modules
module list

# Show what a module does (environment changes)
module show cuda/12.1

# Unload a specific module
module unload cuda/12.1

# Unload ALL modules (clean slate)
module purge

# Save current module set as default
module save

# Restore default modules
module restore
```

**Why modules matter:**
- Different projects need different CUDA versions
- Modules set `PATH`, `LD_LIBRARY_PATH`, and other env vars correctly
- Loading wrong versions causes cryptic errors

### Conda Environment

```bash
# Create environment from environment.yml
conda env create -f environment.yml
#     ↑   ↑      ↑
#     |   |      └── Specification file with dependencies
#     |   └───────── "create" = make new environment
#     └───────────── "env" = environment subcommand

# Create environment with specific name
conda env create -f environment.yml -n my_env_name

# Activate environment (two equivalent methods)
conda activate distillation     # Preferred
source activate distillation    # Works in scripts (more reliable in SLURM)

# Deactivate current environment
conda deactivate

# List all environments
conda env list

# Update environment from modified yml file
conda env update -f environment.yml --prune
#                                    └── Remove packages not in yml

# Export current environment to file
conda env export > environment.yml           # Full export (platform-specific)
conda env export --no-builds > env.yml       # Cross-platform (recommended)
conda env export --from-history > env.yml    # Only explicitly installed packages

# Remove an environment completely
conda env remove -n old_environment

# Clone an environment
conda create --name new_env --clone old_env
```

### Python/Pip in Conda

```bash
# Install a package with pip (inside activated env)
pip install wandb

# Install from requirements.txt
pip install -r requirements.txt

# Install in editable/development mode
pip install -e .

# Check installed packages
pip list
pip show <package>

# Verify GPU is accessible
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0)}')"
```

---

## Understanding SLURM

SLURM (Simple Linux Utility for Resource Management) is the job scheduler. It:
1. Queues jobs from all users
2. Allocates resources (CPUs, GPUs, memory) fairly
3. Runs jobs when resources are available
4. Tracks job history and resource usage

### Key Concepts

| Term | Definition |
|------|------------|
| **Job** | A script/command submitted to run on compute nodes |
| **Node** | A physical server with CPUs, RAM, and optionally GPUs |
| **Partition** | A group of nodes with similar properties (like "gpu", "short") |
| **QOS** | Quality of Service - defines limits and priorities |
| **Core/CPU** | A single processing unit (nodes have many) |
| **Task** | A single process (usually 1 task = 1 Python process) |
| **GRES** | Generic Resource - used for GPUs and other special hardware |

### SLURM Commands Overview

| Command | Purpose |
|---------|---------|
| `sbatch` | Submit a batch job script |
| `srun` | Run a command interactively OR as a job step |
| `squeue` | View queued and running jobs |
| `scancel` | Cancel a job |
| `sinfo` | View cluster/partition status |
| `sacct` | View job accounting/history |
| `scontrol` | View/modify job and node details |
| `sacctmgr` | View account and QOS settings |
| `sprio` | View job priority factors |

---

## Job Submission (sbatch)

### Basic Syntax

```bash
sbatch [options] script.sh [script_arguments]
```

### All SBATCH Directives Explained

Place these at the top of your script (after `#!/bin/bash`):

```bash
#!/bin/bash

#---------------------------------------------------------------------------
# JOB IDENTIFICATION
#---------------------------------------------------------------------------

#SBATCH --job-name=distill
# Short name for the job (appears in squeue)
# Default: script filename
# Tip: Keep it short but descriptive (max ~20 chars shown in squeue)

#SBATCH --account=my_account
# Billing account (if you have multiple)
# Usually not needed for course projects

#---------------------------------------------------------------------------
# PARTITION AND QOS
#---------------------------------------------------------------------------

#SBATCH --partition=gpu
# Which partition (queue) to submit to
# Options: gpu, short, long, courses-gpu, debug, express
# Each partition has different time limits and resources

#SBATCH --qos=gpu
# Quality of Service - controls limits and priority
# Usually matches partition name
# Check your limits: sacctmgr show qos

#---------------------------------------------------------------------------
# RESOURCE REQUESTS
#---------------------------------------------------------------------------

#SBATCH --nodes=1
# Number of nodes (physical machines)
# Default: 1 (usually what you want for DL)
# Multi-node only for distributed training

#SBATCH --ntasks=1
# Number of tasks (processes) to run
# Default: 1
# For single-GPU training: 1
# For multi-GPU with torchrun: still 1 (torchrun spawns workers)

#SBATCH --ntasks-per-node=1
# Tasks per node (for multi-node jobs)
# Usually equals --ntasks when using 1 node

#SBATCH --cpus-per-task=8
# CPU cores allocated to each task
# Rule of thumb: 4-8 CPUs per GPU for data loading
# More CPUs = faster data preprocessing

#SBATCH --mem=64G
# Total memory for the job
# Formats: 64G (gigabytes), 64000M (megabytes), 64000000K (kilobytes)
# Rule of thumb: 8-16GB per GPU, more for large datasets

#SBATCH --mem-per-cpu=8G
# Memory per CPU core (alternative to --mem)
# Use ONE of --mem or --mem-per-cpu, not both

#SBATCH --gres=gpu:h200:1
# Generic resources (GPUs)
# Format: gpu:TYPE:COUNT or gpu:COUNT
# Examples:
#   --gres=gpu:1         # Any 1 GPU
#   --gres=gpu:2         # Any 2 GPUs
#   --gres=gpu:a100:1    # Specifically 1 A100
#   --gres=gpu:h200:2    # Specifically 2 H200s

#---------------------------------------------------------------------------
# TIME LIMITS
#---------------------------------------------------------------------------

#SBATCH --time=08:00:00
# Maximum wall-clock time (job killed if exceeded)
# Formats: MM, MM:SS, HH:MM:SS, D-HH:MM:SS
# Examples:
#   --time=60          # 60 minutes
#   --time=4:00:00     # 4 hours
#   --time=1-12:00:00  # 1 day 12 hours
# Tip: Request slightly more than expected, but not excessively
#      (longer requests = lower priority in queue)

#SBATCH --time-min=01:00:00
# Minimum time (for backfill scheduling)
# Helps scheduler fit your job in gaps

#---------------------------------------------------------------------------
# OUTPUT AND LOGGING
#---------------------------------------------------------------------------

#SBATCH --output=logs/%j.out
# Where to write stdout (terminal output)
# %j = job ID (replaced at runtime)
# %x = job name
# %N = node name
# %a = array task ID
# Examples:
#   logs/%j.out           -> logs/12345.out
#   logs/%x_%j.out        -> logs/distill_12345.out
#   logs/%j_%N.out        -> logs/12345_d4053.out

#SBATCH --error=logs/%j.err
# Where to write stderr (error output)
# Same patterns as --output
# Tip: Can use same file as output: --error=logs/%j.out

#SBATCH --open-mode=append
# How to handle existing output files
# Options: append (add to existing), truncate (overwrite)
# Default: truncate

#SBATCH --mail-user=your.email@northeastern.edu
# Email address for notifications

#SBATCH --mail-type=END,FAIL
# When to send email
# Options: BEGIN, END, FAIL, REQUEUE, ALL, NONE
# Comma-separate multiple: END,FAIL

#---------------------------------------------------------------------------
# SIGNALS AND CHECKPOINTING
#---------------------------------------------------------------------------

#SBATCH --signal=SIGUSR1@120
# Send signal before job timeout
# Format: SIGNAL@SECONDS_BEFORE_END
# SIGUSR1@120 = Send SIGUSR1 signal 120 seconds (2 min) before timeout
# Use this to trigger checkpoint saving before job is killed

#SBATCH --requeue
# Automatically requeue job if node fails
# Useful for long jobs on unreliable hardware

#---------------------------------------------------------------------------
# JOB ARRAYS
#---------------------------------------------------------------------------

#SBATCH --array=0-9
# Run multiple similar jobs with different indices
# Formats:
#   --array=0-9           # Indices 0,1,2,...,9 (10 jobs)
#   --array=1,3,5,7       # Specific indices only
#   --array=0-100%10      # 0-100 but max 10 running at once
#   --array=0-100:2       # 0,2,4,6,...,100 (step by 2)
# Access index in script: $SLURM_ARRAY_TASK_ID

#---------------------------------------------------------------------------
# DEPENDENCIES
#---------------------------------------------------------------------------

#SBATCH --dependency=afterok:12345
# Only start after another job completes
# Types:
#   afterok:JOBID      # After job succeeds (exit code 0)
#   afternotok:JOBID   # After job fails
#   afterany:JOBID     # After job completes (success or fail)
#   after:JOBID        # After job starts
#   singleton          # Only one job with this name runs at a time
# Multiple: --dependency=afterok:123:456:789

#---------------------------------------------------------------------------
# NODE SELECTION
#---------------------------------------------------------------------------

#SBATCH --nodelist=d4053,d4054
# Run ONLY on these specific nodes
# Use for reproducibility or to target specific hardware

#SBATCH --exclude=d4055
# Do NOT run on these nodes
# Use to avoid problematic nodes

#SBATCH --constraint=h200
# Only nodes with this feature
# Features are cluster-specific

#---------------------------------------------------------------------------
# ADVANCED OPTIONS
#---------------------------------------------------------------------------

#SBATCH --exclusive
# Get the entire node (no sharing)
# Use for benchmarking or memory-intensive jobs

#SBATCH --export=ALL
# Which environment variables to export to job
# ALL = all current variables (default)
# NONE = clean environment
# VAR1,VAR2 = only these variables

#SBATCH --propagate=NONE
# Which resource limits to propagate
# NONE recommended to avoid ulimit issues
```

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

# Print job info (helpful for debugging)
echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Started at: $(date)"
echo "========================================="

# Load required modules
module load cuda/12.1
module load anaconda3/2024.06

# Activate conda environment
source activate distillation

# Parse command line arguments with defaults
CONFIG=${1:-configs/default.yaml}   # First arg or default
RUN_NAME=${2:-slurm_run}            # Second arg or default
SEED=${3:-42}                       # Third arg or default

# Create output directories
mkdir -p logs outputs

# Change to project directory (if not already there)
cd /scratch/$USER/Distillation

# Run training
python distill.py \
    --config "$CONFIG" \
    --output_dir "./outputs/$RUN_NAME" \
    --seed "$SEED"

# Capture exit code
EXIT_CODE=$?

echo "========================================="
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================="

exit $EXIT_CODE
```

### Submitting Jobs

```bash
# Basic submission
sbatch script.sh

# With arguments (passed to script as $1, $2, ...)
sbatch script.sh configs/my_config.yaml my_experiment 42

# Override directives from command line (takes precedence over #SBATCH)
sbatch --partition=short --time=01:00:00 script.sh

# Submit with specific GPU type
sbatch --gres=gpu:a100:1 script.sh

# Submit multiple jobs in a loop
for config in configs/ablations/*.yaml; do
    name=$(basename "$config" .yaml)
    sbatch --job-name="$name" script.sh "$config" "$name"
done
```

---

## Interactive Jobs (srun)

`srun` runs commands on compute nodes. Two main uses:
1. **Interactive sessions** - Get a shell on a compute node
2. **Job steps** - Run commands within a batch job

### Interactive Sessions

```bash
# Basic interactive GPU session
srun --partition=gpu --gres=gpu:1 --time=01:00:00 --pty bash
#                                                 └── --pty: Pseudo-terminal
#                                                     Required for interactive shell

# Full options for interactive work
srun \
    --partition=gpu \           # GPU partition
    --gres=gpu:h200:1 \         # 1 H200 GPU
    --time=04:00:00 \           # 4 hour limit
    --mem=64G \                 # 64GB RAM
    --cpus-per-task=8 \         # 8 CPU cores
    --job-name=interactive \    # Job name
    --pty bash                  # Interactive bash shell
```

**Once inside the interactive session:**
```bash
# You're now on a compute node with a GPU
nvidia-smi                      # Check GPU
module load cuda/12.1 anaconda3/2024.06
source activate distillation
python distill.py --max_steps 100  # Test your code
exit                            # End session (releases resources)
```

### Running Commands Directly

```bash
# Run a single command (non-interactive)
srun --partition=gpu --gres=gpu:1 nvidia-smi

# Run Python script
srun --partition=gpu --gres=gpu:1 --time=00:30:00 \
    python distill.py --config config.yaml --max_steps 100

# Multiple commands with bash -c
srun --partition=gpu --gres=gpu:1 \
    bash -c "nvidia-smi && python -c 'import torch; print(torch.cuda.is_available())'"
```

### srun vs sbatch

| Aspect | srun | sbatch |
|--------|------|--------|
| Execution | Immediate (waits in queue) | Background (returns immediately) |
| Output | Displayed in terminal | Written to file |
| Use case | Testing, debugging, quick jobs | Production training runs |
| Interactivity | Can be interactive (--pty) | Always non-interactive |
| Job script | Not needed | Required |

---

## Job Monitoring

### squeue - View Job Queue

```bash
# Basic: your jobs
squeue -u $USER

# All jobs (cluster-wide)
squeue

# Jobs on specific partition
squeue -p gpu

# Custom format with specific columns
squeue -u $USER -o "%.10i %.20j %.8T %.10M %.6D %R"
#                    │     │     │     │     │   └── Reason/NodeList
#                    │     │     │     │     └────── Nodes
#                    │     │     │     └──────────── Time
#                    │     │     └────────────────── State
#                    │     └──────────────────────── Name (20 chars)
#                    └────────────────────────────── JobID (10 chars)
#                    (%.Ni = N chars, right-aligned)

# Full format options
squeue -u $USER -o "%.10i %.9P %.20j %.8u %.2t %.10M %.10l %.6D %.4C %R"
#                          │           │   │       │           │
#                          │           │   │       │           └── CPUs
#                          │           │   │       └────────────── Time Limit
#                          │           │   └────────────────────── State (short)
#                          │           └────────────────────────── User
#                          └────────────────────────────────────── Partition

# Show estimated start time for pending jobs
squeue -u $USER --start

# Watch queue (refresh every 5 seconds)
watch -n 5 'squeue -u $USER'

# Long format (shows all details)
squeue -u $USER -l
```

#### squeue Output Columns

| Column | Long Name | Description |
|--------|-----------|-------------|
| `JOBID` | Job ID | Unique identifier (use for scancel, sacct) |
| `PARTITION` | Partition | Queue the job is in (gpu, short, etc.) |
| `NAME` | Job Name | From --job-name or script name |
| `USER` | Username | Who submitted the job |
| `ST` | State | Current job state (see below) |
| `TIME` | Time | Time the job has been running |
| `NODES` | Nodes | Number of nodes allocated |
| `NODELIST(REASON)` | Node/Reason | Running: node name; Pending: why waiting |

#### Job States

| State | Code | Description |
|-------|------|-------------|
| PENDING | PD | Waiting in queue for resources |
| RUNNING | R | Currently executing |
| COMPLETING | CG | Job finishing, cleaning up |
| COMPLETED | CD | Finished successfully |
| FAILED | F | Exited with non-zero code |
| TIMEOUT | TO | Killed for exceeding time limit |
| CANCELLED | CA | Cancelled by user or admin |
| NODE_FAIL | NF | Node failed during job |
| PREEMPTED | PR | Killed to make room for higher priority job |
| SUSPENDED | S | Job paused (rare) |

#### Pending Reasons

| Reason | Meaning | What to Do |
|--------|---------|------------|
| `(Priority)` | Lower priority than other jobs | Wait; jobs ahead of you |
| `(Resources)` | Requested resources not available | Wait; resources will free up |
| `(QOSMaxJobsPerUserLimit)` | Hit your concurrent job limit | Wait for running jobs to finish, or cancel some |
| `(QOSMaxSubmitJobsPerUserLimit)` | Hit your submitted job limit | Cancel some pending jobs |
| `(ReqNodeNotAvail)` | Requested specific node is busy/down | Wait, or remove node constraint |
| `(Dependency)` | Waiting for dependent job to complete | Wait for other job |
| `(DependencyNeverSatisfied)` | Dependent job failed | Cancel and resubmit without dependency |
| `(PartitionDown)` | Partition is offline | Try different partition |
| `(PartitionNodeLimit)` | Partition node limit reached | Wait |
| `(BeginTime)` | Job has a future start time | Wait for scheduled time |
| `(JobHeldUser)` | You put job on hold | Release with: scontrol release JOBID |
| `(JobHeldAdmin)` | Admin put job on hold | Contact admin |

### sinfo - View Cluster Status

```bash
# Basic partition summary
sinfo

# Specific partition
sinfo -p gpu

# Node-centric view (one line per node)
sinfo -N

# Long format with all details
sinfo -N -l

# Custom format
sinfo -p gpu -o "%N %P %G %T %C %m"
#               │  │  │  │  │  └── Memory
#               │  │  │  │  └───── CPUs (A/I/O/T = Allocated/Idle/Other/Total)
#               │  │  │  └──────── State
#               │  │  └─────────── GRES (GPUs)
#               │  └────────────── Partition
#               └──────────────── Node Name

# Show only GPU nodes with specific format
sinfo -p gpu -N -o "%N %G %t %C %m %f"
#                              │  └── Features
#                              └───── Memory

# Summarize by state
sinfo -p gpu -o "%P %a %D %t"
#               │  │  │  └── State
#               │  │  └───── Node count
#               │  └──────── Availability
#               └─────────── Partition

# Node state meanings in CPUS column: A/I/O/T
# A = Allocated (in use)
# I = Idle (available)
# O = Other (draining, down, etc.)
# T = Total
```

#### Node States

| State | Symbol | Description |
|-------|--------|-------------|
| `idle` | `~` | Node is free, ready for jobs |
| `allocated`/`alloc` | `#` | All resources in use |
| `mixed`/`mix` | `@` | Some resources in use, some free |
| `down` | `*` | Node offline (hardware issue, maintenance) |
| `drained`/`drain` | `%` | No new jobs; existing jobs can finish |
| `draining`/`drng` | `$` | Being drained; has running jobs |
| `reserved`/`resv` | `&` | Reserved for specific users/jobs |
| `maintenance`/`maint` | `!` | Under maintenance |

### sacct - Job Accounting History

```bash
# Your recent jobs (today)
sacct

# Specify user explicitly
sacct -u $USER

# Specific job by ID
sacct -j 12345

# Jobs from date range
sacct -u $USER -S 2024-01-15           # Since Jan 15
sacct -u $USER -S 2024-01-15 -E 2024-01-20  # Jan 15-20
sacct -u $USER -S $(date -d "7 days ago" +%Y-%m-%d)  # Last 7 days

# Custom format (recommended for useful info)
sacct -j 12345 --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,AllocCPUS,AllocGRES

# Common format for checking why jobs failed
sacct -j 12345 --format=JobID,JobName,State,ExitCode,DerivedExitCode,Elapsed

# Check memory usage (important for OOM debugging)
sacct -j 12345 --format=JobID,MaxRSS,MaxVMSize,ReqMem

# Show only main job (not job steps)
sacct -j 12345 --noheader -X

# All completed jobs by name pattern
sacct -u $USER --name="distill%" --state=COMPLETED
```

#### Useful sacct Format Fields

| Field | Description |
|-------|-------------|
| `JobID` | Job identifier (includes step suffixes like .batch, .0) |
| `JobName` | Job name |
| `Partition` | Partition used |
| `Account` | Account charged |
| `State` | Final state (COMPLETED, FAILED, TIMEOUT, etc.) |
| `ExitCode` | Exit code (0:0 = success, X:0 = exit code X, 0:Y = signal Y) |
| `DerivedExitCode` | Exit code considering all steps |
| `Elapsed` | Wall-clock time used |
| `TotalCPU` | Total CPU time used |
| `CPUTime` | CPU time = Elapsed × AllocCPUS |
| `MaxRSS` | Maximum RAM used (resident set size) |
| `MaxVMSize` | Maximum virtual memory |
| `ReqMem` | Requested memory |
| `AllocCPUS` | CPUs allocated |
| `AllocGRES` | GPUs allocated |
| `NodeList` | Node(s) used |

### scontrol - View/Modify Jobs and Nodes

```bash
# Full details about a job
scontrol show job 12345

# Specific job fields
scontrol show job 12345 | grep -E "(JobId|JobState|StartTime|EndTime|NodeList)"

# Node details
scontrol show node d4053

# Partition details
scontrol show partition gpu

# Modify a pending job (can't modify running jobs)
scontrol update JobId=12345 TimeLimit=04:00:00
scontrol update JobId=12345 Partition=short
scontrol update JobId=12345 NumCPUs=4

# Hold a job (prevent from starting)
scontrol hold 12345

# Release a held job
scontrol release 12345

# Requeue a failed job (restart from beginning)
scontrol requeue 12345
```

### scancel - Cancel Jobs

```bash
# Cancel specific job
scancel 12345

# Cancel multiple jobs
scancel 12345 12346 12347

# Cancel all your jobs
scancel -u $USER

# Cancel all pending jobs (keep running ones)
scancel -u $USER --state=PENDING
# Or: scancel -u $USER -t PENDING

# Cancel all running jobs (keep pending ones)
scancel -u $USER --state=RUNNING

# Cancel by job name (pattern matching)
scancel -u $USER --name=distill
scancel -u $USER --name="distill%"    # Wildcard

# Cancel by partition
scancel -u $USER --partition=gpu

# Cancel with signal (for graceful shutdown)
scancel --signal=SIGTERM 12345      # Send SIGTERM first
scancel --signal=SIGUSR1 12345      # Custom signal for checkpointing

# Cancel array job tasks
scancel 12345_5                      # Cancel task 5 of array
scancel 12345_[1-5]                  # Cancel tasks 1-5

# Cancel interactively (asks confirmation)
scancel -i -u $USER
```

### sprio - View Job Priority

```bash
# See priority factors for your jobs
sprio -u $USER

# Detailed priority breakdown
sprio -u $USER -l

# Specific job
sprio -j 12345

# Compare with others
sprio --partition=gpu | head -20
```

Priority factors (higher = runs sooner):
- **Age** - How long job has been waiting
- **FairShare** - Based on your recent usage vs allocation
- **JobSize** - Smaller jobs may get priority
- **Partition** - Different partitions have different weights
- **QOS** - Quality of Service settings

---

## GPU Resources

### Available GPUs on Explorer

| GPU Type | Nodes | GPUs/Node | Total | VRAM | Compute Cap. | Best For |
|----------|-------|-----------|-------|------|--------------|----------|
| H200 | d4052-4055 | 8 | 32 | 80GB HBM3 | sm_90 | Large models, fastest training |
| A100-80GB | Various | 4-8 | ~50 | 80GB HBM2e | sm_80 | Large batch training |
| A100-40GB | Various | 4-8 | ~50 | 40GB HBM2e | sm_80 | Standard DL training |
| V100 | Various | 4-8 | ~50 | 32GB HBM2 | sm_70 | Medium workloads |
| P100 | Various | 2-4 | ~30 | 16GB HBM2 | sm_60 | Small models, testing |

### GPU Request Syntax

```bash
# Generic Resource (GRES) format
--gres=gpu:COUNT              # Any GPU type
--gres=gpu:TYPE:COUNT         # Specific GPU type

# Examples
--gres=gpu:1                  # 1 GPU (any type)
--gres=gpu:2                  # 2 GPUs (any type, same node)
--gres=gpu:a100:1             # 1 A100
--gres=gpu:h200:1             # 1 H200
--gres=gpu:v100:2             # 2 V100s
--gres=gpu:a100:4             # 4 A100s (multi-GPU training)
```

### Check GPU Availability

```bash
# See GPU nodes and their state
sinfo -p gpu -o "%20N %10G %10T %10c %10m"
#               Node     GPUs  State  CPUs   Memory

# See which GPUs are free (node by node)
sinfo -p gpu -N -o "%N %G %t %C %m"
#                  │  │  │  │  └── Memory
#                  │  │  │  └───── CPUs A/I/O/T
#                  │  │  └──────── State (short)
#                  │  └─────────── GRES (GPUs)
#                  └────────────── Node name

# Check specific node type (e.g., H200s)
sinfo -N -p gpu | grep d405

# See all jobs on H200 nodes
squeue -p gpu -w d4052,d4053,d4054,d4055

# Detailed node info
scontrol show node d4053 | grep -E "(NodeName|Gres|State|CPUAlloc|RealMemory)"
```

### GPU Commands Inside Jobs

```bash
# Which GPU(s) are assigned to your job
echo $CUDA_VISIBLE_DEVICES
# Output: 0 (or 0,1 for multi-GPU, etc.)

# GPU information and current usage
nvidia-smi

# Watch GPU usage (updates every second)
watch -n 1 nvidia-smi

# Query specific info
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv

# GPU memory usage only
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Continuous logging (every second)
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 1

# CUDA version
nvcc --version

# PyTorch CUDA check
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

---

## Quality of Service (QOS)

QOS controls job limits and priorities.

### View QOS Limits

```bash
# See all QOS settings
sacctmgr show qos format=Name,MaxJobsPU,MaxSubmitPU,MaxWall,Priority

# Your specific associations
sacctmgr show assoc user=$USER format=User,Account,QOS,MaxJobs,MaxSubmit
```

### Common QOS on Explorer

| QOS | MaxJobsPerUser | MaxSubmitPerUser | Max Time | Notes |
|-----|----------------|------------------|----------|-------|
| `gpu` | 4 | 8 | 24 hours | Standard GPU jobs |
| `gpu-short` | 2 | 8 | 4 hours | Quick GPU jobs |
| `short` | 50 | - | 4 hours | CPU only |
| `long` | 25 | - | 7 days | Extended runs |
| `courses-gpu` | 2 | - | 8 hours | Course projects |
| `debug` | 10 | - | 1 hour | Testing |

### QOS Implications

- **MaxJobsPerUser**: How many jobs can RUN simultaneously
- **MaxSubmitPerUser**: How many jobs can be in queue total (running + pending)
- If you hit limits, new jobs show `(QOSMaxJobsPerUserLimit)` or `(QOSMaxSubmitJobsPerUserLimit)`

---

## SLURM Environment Variables

These are automatically set inside your job:

### Job Information

| Variable | Description | Example |
|----------|-------------|---------|
| `SLURM_JOB_ID` | Unique job identifier | `12345` |
| `SLURM_JOB_NAME` | Job name (from --job-name) | `distill` |
| `SLURM_SUBMIT_DIR` | Directory where sbatch was run | `/scratch/user/project` |
| `SLURM_JOB_NODELIST` | Nodes allocated | `d4053` |
| `SLURM_NODELIST` | Same as above | `d4053` |
| `SLURM_JOB_NUM_NODES` | Number of nodes | `1` |
| `SLURM_NNODES` | Same as above | `1` |

### Task Information

| Variable | Description | Example |
|----------|-------------|---------|
| `SLURM_NTASKS` | Number of tasks | `1` |
| `SLURM_CPUS_PER_TASK` | CPUs per task | `8` |
| `SLURM_CPUS_ON_NODE` | Total CPUs on node | `64` |
| `SLURM_PROCID` | Task rank (0 to ntasks-1) | `0` |
| `SLURM_LOCALID` | Task rank on current node | `0` |

### GPU/Resource Information

| Variable | Description | Example |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU indices visible to CUDA | `0` or `0,1,2,3` |
| `SLURM_GPUS` | Total GPUs allocated | `1` |
| `SLURM_GPUS_ON_NODE` | GPUs on current node | `1` |
| `SLURM_JOB_GPUS` | GPU indices (different format) | `0` |

### Array Job Information

| Variable | Description | Example |
|----------|-------------|---------|
| `SLURM_ARRAY_JOB_ID` | Parent array job ID | `12345` |
| `SLURM_ARRAY_TASK_ID` | This task's index | `3` |
| `SLURM_ARRAY_TASK_COUNT` | Total tasks in array | `10` |
| `SLURM_ARRAY_TASK_MIN` | Minimum index | `0` |
| `SLURM_ARRAY_TASK_MAX` | Maximum index | `9` |

### Usage in Scripts

```bash
#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --array=0-4

# Use job ID for unique output directory
OUTPUT_DIR="outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# Use array index to select hyperparameters
LEARNING_RATES=(1e-5 2e-5 5e-5 1e-4 2e-4)
LR=${LEARNING_RATES[$SLURM_ARRAY_TASK_ID]}

# Use array index as seed
SEED=$((42 + SLURM_ARRAY_TASK_ID))

echo "Job $SLURM_JOB_ID, Task $SLURM_ARRAY_TASK_ID: LR=$LR, Seed=$SEED"

python distill.py --lr $LR --seed $SEED --output_dir "$OUTPUT_DIR"
```

---

## Debugging

### View Job Logs

```bash
# stdout (normal output)
cat logs/12345.out

# stderr (errors)
cat logs/12345.err

# Follow log in real-time (while job runs)
tail -f logs/12345.out

# Last N lines
tail -100 logs/12345.out

# First N lines (see startup messages)
head -50 logs/12345.out

# Search for patterns
grep -i "error" logs/12345.err
grep -i "cuda" logs/12345.out
grep "loss" logs/12345.out | tail -20

# View both files together
cat logs/12345.{out,err} | less
```

### Common Errors and Solutions

#### 1. Job Fails Immediately (Exit Code 1)

```bash
# Check stderr for the actual error
cat logs/<jobid>.err

# Common causes:
# - Module not loaded
# - Conda environment not activated
# - Python syntax error
# - Missing file/directory
# - Permission denied
```

#### 2. CUDA Out of Memory

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**
```python
# Reduce batch size
batch_size = batch_size // 2

# Use gradient accumulation instead
--gradient_accumulation_steps 2

# Enable mixed precision (fp16)
--fp16 True

# Clear cache periodically
torch.cuda.empty_cache()

# Use gradient checkpointing
model.gradient_checkpointing_enable()
```

#### 3. CPU Out of Memory (OOM Killer)

```
slurmstepd: error: Detected 1 oom-kill event(s)
```

**Solutions:**
```bash
# Request more memory
#SBATCH --mem=128G

# Reduce number of data loader workers
--dataloader_num_workers 2

# Use memory-efficient data loading
--pin_memory False
```

#### 4. Job Killed by Timeout

```
slurmstepd: *** JOB 12345 ON d4053 CANCELLED AT ... DUE TO TIME LIMIT ***
```

**Solutions:**
```bash
# Request more time
#SBATCH --time=12:00:00

# Implement checkpointing to resume
#SBATCH --signal=SIGUSR1@120  # Signal 2 min before kill

# Handle signal in Python:
import signal
def checkpoint_handler(signum, frame):
    save_checkpoint()
    sys.exit(0)
signal.signal(signal.SIGUSR1, checkpoint_handler)
```

#### 5. Module/Environment Errors

```
module: command not found
# or
conda: command not found
```

**Solution - add to script:**
```bash
# Initialize shell for modules
source /etc/profile.d/modules.sh

# For conda in non-interactive shells
source /shared/centos7/anaconda3/2024.06/etc/profile.d/conda.sh
```

#### 6. CUDA Version Mismatch

```
RuntimeError: CUDA error: no kernel image is available for execution
```

**Solutions:**
```bash
# Check versions match
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Load correct CUDA module
module load cuda/12.1  # Match your PyTorch CUDA version
```

### SSH into Running Job

```bash
# Find which node your job is on
squeue -u $USER
# Note the NODELIST (e.g., d4053)

# SSH to that node (only works while job is running)
ssh d4053

# Find your processes
ps aux | grep $USER

# Check GPU usage
nvidia-smi

# Check memory
free -h

# Exit when done
exit
```

### Debug Interactively

```bash
# Get interactive session with same resources
srun --partition=gpu --gres=gpu:1 --mem=64G --cpus-per-task=8 --time=01:00:00 --pty bash

# Once on compute node:
module load cuda/12.1 anaconda3/2024.06
source activate distillation

# Run script manually to see full errors
python distill.py --config config.yaml

# Or use debugger
python -m pdb distill.py --config config.yaml
# Commands: n (next), s (step), c (continue), q (quit), p variable (print)

# Or use IPython for interactive debugging
python -c "
import train
# ... test components
"
```

### Memory Profiling

```bash
# Check job's actual memory usage after completion
sacct -j 12345 --format=JobID,JobName,MaxRSS,MaxVMSize,ReqMem,State

# MaxRSS format: e.g., 32456K, 15G
# Compare MaxRSS to ReqMem - if close, increase --mem

# Monitor live memory (on compute node)
watch -n 1 free -h

# GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1

# Python memory profiling
pip install memory_profiler
python -m memory_profiler distill.py
```

---

## Data Transfer

### Between Local Machine and Explorer

```bash
# Upload files (from local machine)
scp file.txt username@login.explorer.northeastern.edu:/scratch/username/
scp -r folder/ username@login.explorer.northeastern.edu:/scratch/username/

# Download files (to local machine)
scp username@login.explorer.northeastern.edu:/scratch/username/results.tar.gz .
scp -r username@login.explorer.northeastern.edu:/scratch/username/outputs/ ./local_outputs/

# rsync (better for large transfers, resumable)
rsync -avz --progress folder/ username@login.explorer.northeastern.edu:/scratch/username/folder/
rsync -avz --progress username@login.explorer.northeastern.edu:/scratch/username/outputs/ ./outputs/

# rsync with exclusions
rsync -avz --progress --exclude='*.pyc' --exclude='__pycache__' \
    ./ username@login.explorer.northeastern.edu:/scratch/username/project/
```

### Large Dataset Transfer

```bash
# For very large files, use Globus (web-based, more reliable)
# https://www.globus.org

# Or use screen/tmux for long transfers
screen -S transfer
rsync -avz --progress large_dataset/ explorer:/scratch/user/data/
# Ctrl+A, D to detach
# screen -r transfer to reattach
```

### Git Operations on Explorer

```bash
# Clone (HTTPS - may need to enter password)
git clone https://github.com/user/repo.git

# Clone (SSH - needs SSH key setup)
git clone git@github.com:user/repo.git

# If using SSH agent forwarding (ssh -A):
# Your local SSH keys are available on Explorer
git push origin main

# Pull latest changes
git pull

# Stage and commit changes
git add -A
git commit -m "Results from experiment"
git push
```

---

## Best Practices

### 1. Project Organization

```
/scratch/<user>/project/
├── configs/               # YAML configuration files
│   └── ablations/         # Ablation study configs
├── scripts/               # SLURM job scripts
│   └── run_training.sh
├── logs/                  # Job output logs (gitignore)
├── outputs/               # Model checkpoints (gitignore)
├── src/                   # Source code
├── tests/                 # Unit tests
├── environment.yml        # Conda environment
├── pyproject.toml         # Project metadata
├── .gitignore
└── README.md
```

### 2. .gitignore for HPC

```gitignore
# SLURM logs
logs/
*.out
*.err

# Model outputs
outputs/
checkpoints/
*.pt
*.pth
*.ckpt

# Data
data/
*.csv
*.parquet

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# Environment
.env
*.egg-info/

# Misc
.DS_Store
*.swp
```

### 3. Checkpointing for Long Jobs

```bash
# In SBATCH script, handle preemption signal
#SBATCH --signal=SIGUSR1@120

# Your code should:
# 1. Save checkpoints regularly
# 2. Handle SIGUSR1 to save before timeout
# 3. Automatically resume from latest checkpoint
```

### 4. Resource Estimation

| Model Size | GPU Memory | Recommended GPU | Time/Epoch (MNLI) |
|------------|------------|-----------------|-------------------|
| ~66M (DistilBERT) | ~2GB | P100/V100 | ~10-15 min |
| ~110M (BERT) | ~4GB | V100/A100 | ~15-20 min |
| ~355M (BERT-Large) | ~12GB | A100 | ~45-60 min |
| ~1B | ~20GB | A100-40GB | hours |
| ~3B | ~40GB | A100-80GB/H200 | hours |

### 5. Testing Before Production Runs

```bash
# Always test with minimal steps first
srun --partition=gpu --gres=gpu:1 --time=00:30:00 --pty bash

# Inside interactive session
python distill.py --config config.yaml --max_steps 100
# Check for errors, memory usage, output format

# Then submit full job
sbatch scripts/run_training.sh
```

### 6. Multiple Experiment Management

```bash
# Submit multiple configs systematically
for config in configs/ablations/*.yaml; do
    name=$(basename "$config" .yaml)
    echo "Submitting $name"
    sbatch --job-name="$name" scripts/run_training.sh "$config" "$name"
    sleep 1  # Avoid overwhelming the scheduler
done

# Track experiments with W&B
# Each run gets unique ID, stored in run_metadata.json
# Resume automatically finds correct W&B run
```

### 7. Cleanup

```bash
# Remove old checkpoints (keep last 3)
# This is done by save_total_limit in TrainingArguments

# Clean old logs (older than 30 days)
find logs/ -name "*.out" -mtime +30 -delete
find logs/ -name "*.err" -mtime +30 -delete

# Check disk usage
du -sh /scratch/$USER/*
df -h /scratch

# Find large files
find /scratch/$USER -size +1G -exec ls -lh {} \;
```

---

## Quick Reference Card

### Essential Commands

| Task | Command |
|------|---------|
| Submit job | `sbatch script.sh` |
| Submit with args | `sbatch script.sh config.yaml run_name` |
| Interactive GPU | `srun --partition=gpu --gres=gpu:1 --time=01:00:00 --pty bash` |
| View your jobs | `squeue -u $USER` |
| Detailed job view | `squeue -u $USER -o "%.10i %.20j %.8T %.10M %.6D %R"` |
| Cancel job | `scancel JOBID` |
| Cancel all jobs | `scancel -u $USER` |
| View cluster | `sinfo -p gpu` |
| Job history | `sacct -u $USER` |
| Job details | `scontrol show job JOBID` |
| View logs | `tail -f logs/JOBID.out` |
| Check GPU | `nvidia-smi` |
| Check QOS limits | `sacctmgr show qos format=Name,MaxJobsPU,MaxSubmitPU` |

### Module Setup

```bash
module load cuda/12.1 anaconda3/2024.06 && source activate distillation
```

### GPU Types

| Type | Request | VRAM | Speed |
|------|---------|------|-------|
| H200 | `--gres=gpu:h200:1` | 80GB | Fastest |
| A100 | `--gres=gpu:a100:1` | 40/80GB | Fast |
| V100 | `--gres=gpu:v100:1` | 32GB | Medium |
| P100 | `--gres=gpu:p100:1` | 16GB | Slower |

### Partition Limits

| Partition | Max Time | Max Jobs | Use Case |
|-----------|----------|----------|----------|
| `gpu` | 24h | 4 running | Standard GPU |
| `gpu-short` | 4h | 2 running | Quick tests |
| `short` | 4h | 50 | CPU jobs |
| `long` | 7d | 25 | Extended |
| `courses-gpu` | 8h | 2 | Courses |

---

## References

- [Explorer Documentation](https://rc-docs.northeastern.edu/)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
- [SLURM sbatch Manual](https://slurm.schedmd.com/sbatch.html)
- [SLURM srun Manual](https://slurm.schedmd.com/srun.html)
- [Research Computing Help](mailto:rchelp@northeastern.edu)
