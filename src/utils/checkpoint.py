"""Training state management for SLURM job chaining."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


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
            with open(self.state_file, 'r') as f:
                return json.load(f)

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

    def get_resume_checkpoint(self, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Find the latest checkpoint to resume from.

        Args:
            output_dir: Directory to search for checkpoints (defaults to self.output_dir)

        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)

        checkpoints = list(output_dir.glob("checkpoint-*"))
        if not checkpoints:
            return None

        # Sort by step number and return the latest
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
