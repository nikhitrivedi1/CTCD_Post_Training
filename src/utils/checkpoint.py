"""Training state management for SLURM job chaining."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

LATEST_CHECKPOINT_FILE = "latest_checkpoint"


class TrainingState:
    """
    Manages training state across SLURM job restarts.

    Tracks completed epochs/steps and determines if training should continue.
    Essential for handling SLURM's 8-hour job limits through auto-requeue.
    """

    def __init__(self, output_dir: str):
        """
        Initialize training state manager.

        Args:
            output_dir: Directory where state file will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.output_dir / "training_state.json"
        self.state = self._load()

    def _load(self) -> Dict[str, Any]:
        """
        Load existing state from file or create new state.

        Returns:
            Dictionary containing training state
        """
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except ValueError:
                logger.warning(
                    "Corrupt training_state.json in %s — starting fresh",
                    self.output_dir,
                )

        return {
            "completed_epochs": 0,
            "completed_steps": 0,
            "target_epochs": None,
            "target_steps": None,
            "jobs": []
        }

    def save(self, epoch: float, step: int):
        """
        Save current training progress.

        Args:
            epoch: Current epoch (can be fractional)
            step: Current global step
        """
        self.state["completed_epochs"] = int(epoch)
        self.state["completed_steps"] = step
        self.state["jobs"].append({
            "job_id": os.environ.get("SLURM_JOB_ID", "local"),
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "step": step
        })

        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def is_complete(self) -> bool:
        """
        Check if training has completed based on target epochs/steps.

        Returns:
            True if training is complete, False otherwise
        """
        if self.state["target_epochs"] is not None:
            return self.state["completed_epochs"] >= self.state["target_epochs"]
        if self.state["target_steps"] is not None:
            return self.state["completed_steps"] >= self.state["target_steps"]
        return False

    def is_pipeline_complete(self) -> bool:
        """
        Check if the full pipeline (training + post-eval) has completed.

        Set only after post-training evaluation finishes successfully.
        If training completes but Python is killed during post-eval,
        this remains False so the resubmitted job re-runs post-eval.

        Returns:
            True if full pipeline is complete, False otherwise
        """
        return self.state.get("pipeline_complete", False)

    def mark_pipeline_complete(self):
        """Mark the full pipeline as complete (training + post-eval done)."""
        self.state["pipeline_complete"] = True
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def get_resume_checkpoint(self, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Find the latest safely-completed checkpoint to resume from.

        Reads the `latest_checkpoint` file, which is only written after a
        checkpoint is fully saved. Falls back to globbing checkpoint dirs
        if the file doesn't exist (backward compat).

        Args:
            output_dir: Directory to search for checkpoints (defaults to self.output_dir)

        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)

        # Prefer latest_checkpoint file (atomic, corruption-safe)
        latest_file = output_dir / LATEST_CHECKPOINT_FILE
        if latest_file.exists():
            raw_path = latest_file.read_text().strip()
            if raw_path:
                checkpoint_path = Path(raw_path)
                # Try as-is first (may be absolute or already include output_dir)
                if checkpoint_path.exists() and checkpoint_path.is_dir():
                    logger.info(f"Resuming from latest_checkpoint: {checkpoint_path}")
                    return str(checkpoint_path)
                # Try relative to output_dir
                if not checkpoint_path.is_absolute():
                    resolved = (output_dir / checkpoint_path).resolve()
                    if resolved.exists() and resolved.is_dir():
                        logger.info(f"Resuming from latest_checkpoint: {resolved}")
                        return str(resolved)
            logger.warning(
                f"latest_checkpoint points to {raw_path} but it doesn't exist"
            )

        # Fallback: glob for checkpoint dirs (backward compat)
        checkpoints = []
        for p in output_dir.glob("checkpoint-*"):
            if not p.is_dir():
                continue
            try:
                int(p.name.split("-")[1])
                checkpoints.append(p)
            except (ValueError, IndexError):
                continue
        if not checkpoints:
            return None

        latest = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
        return str(latest)

    def set_targets(self, epochs: Optional[int] = None, steps: Optional[int] = None):
        """
        Set target epochs or steps for training completion.

        Args:
            epochs: Target number of epochs
            steps: Target number of steps
        """
        if epochs is not None:
            self.state["target_epochs"] = epochs
        if steps is not None:
            self.state["target_steps"] = steps

        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)


def mark_checkpoint_complete(checkpoint_dir: str):
    """
    Record a checkpoint as the latest safely-completed one.

    Called after HF Trainer finishes writing all checkpoint files. Writes
    the checkpoint path to `latest_checkpoint` in the parent output dir.
    If the process is killed mid-checkpoint-save, this file still points
    to the previous good checkpoint.

    Args:
        checkpoint_dir: Path to the completed checkpoint directory
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = checkpoint_dir.parent
    latest_file = output_dir / LATEST_CHECKPOINT_FILE

    # Atomic write: write to temp file, then rename
    tmp = latest_file.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(str(checkpoint_dir))
    os.replace(tmp, latest_file)

    logger.info(f"Marked latest checkpoint: {checkpoint_dir.name}")
