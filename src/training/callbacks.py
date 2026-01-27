"""Callbacks for SLURM signal handling and job chaining."""

import signal

from transformers import TrainerCallback


class SLURMSignalCallback(TrainerCallback):
    """
    Handle SLURM signals for graceful job termination.

    SLURM sends SIGUSR1 before the job time limit is reached (configured with
    --signal=SIGUSR1@120 for 2 minutes before), giving us time to save state.
    This callback catches the signal and triggers checkpoint saving.

    Critical for handling 8-hour job limits on Explorer HPC cluster.
    """

    def __init__(self, trainer):
        """
        Initialize signal handler.

        Args:
            trainer: HuggingFace Trainer instance
        """
        self.trainer = trainer
        self.caught_signal = False

        # Register signal handlers
        signal.signal(signal.SIGUSR1, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

        print("SLURM signal handler registered (SIGUSR1, SIGTERM)")

    def _handler(self, signum, frame):
        """
        Signal handler that triggers checkpoint saving.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        self.caught_signal = True
        print(f"\n{'=' * 60}")
        print(f"Caught signal {signum} - saving checkpoint...")
        print(f"{'=' * 60}\n")

        # Save model and trainer state
        self.trainer.save_model()
        self.trainer.save_state()

        print("Checkpoint saved successfully")

    def on_step_end(self, args, state, control, **kwargs):
        """
        Check if signal was caught and stop training.

        Called after each training step.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            **kwargs: Additional arguments

        Returns:
            Updated control
        """
        if self.caught_signal:
            control.should_save = True
            control.should_training_stop = True
            print("Training will stop at the end of this step")

        return control
