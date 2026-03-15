"""Callbacks for training: signal handling, metrics logging, teacher eval, and more."""

import logging
import os
import signal
from typing import List, Optional

import torch
from transformers import TrainerCallback

from src.utils.checkpoint import mark_checkpoint_complete

logger = logging.getLogger(__name__)


class SLURMSignalCallback(TrainerCallback):
    """
    Handle SLURM signals for graceful job termination.

    The shell forwards SIGUSR1 to Python (via B:SIGUSR1 + _FORGE_CHILD_PID).
    This callback catches the signal and tells the training loop to save a
    checkpoint and stop. The actual checkpoint is saved by HF Trainer's normal
    save path (triggered by should_save=True), not in the signal handler.

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

        logger.info("SLURM signal handler registered (SIGUSR1, SIGTERM)")

    def _handler(self, signum, frame):
        """
        Signal handler that flags training for graceful stop.

        Does NOT save directly (risky in signal context). Instead sets a flag
        that on_step_end checks to trigger should_save + should_training_stop,
        letting HF Trainer save through its normal (safe) checkpoint path.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        self.caught_signal = True
        logger.warning("=" * 60)
        logger.warning(f"Caught signal {signum} - will save and stop after current step")
        logger.warning("=" * 60)

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
            logger.info("Training will stop at the end of this step")

        return control


class CheckpointMarkerCallback(TrainerCallback):
    """
    Mark checkpoints as safely completed after all files are written.

    Writes a `latest_checkpoint` file pointing to the most recently
    completed checkpoint. If the process is killed mid-checkpoint-save,
    this file still points to the previous good checkpoint, preventing
    resume from a corrupt/partial checkpoint.

    Add this callback LAST so it runs after all other on_save hooks
    (e.g. TeacherEvalCallback saving the teacher model).
    """

    def on_save(self, args, state, control, **kwargs):
        """
        Mark the checkpoint as complete after HF Trainer finishes saving.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
        """
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        mark_checkpoint_complete(checkpoint_dir)


class StepMetricsCallback(TrainerCallback):
    """
    Log training metrics to Python logger on each logging step.

    HF Trainer's default ProgressCallback writes metrics as raw dict dumps via
    tqdm.write(). This callback logs them through the Python logger in a clean,
    grep-friendly format so they appear in output.log alongside other log lines.

    Logs two lines per step:
      - step=N epoch=E lr=L loss=V (core training stats)
      - reward=R reward_std=S edit_similarity=V ... (per-reward breakdown, GRPO only)
    """

    # Metrics to always show first (in order)
    _CORE_KEYS = ["loss", "learning_rate", "grad_norm"]
    # Prefixes to group into a second "rewards" line
    _REWARD_PREFIX = "rewards/"

    @staticmethod
    def _fmt(value):
        """Format a numeric value, using scientific notation for very small values."""
        if isinstance(value, float):
            if 0 < abs(value) < 1e-4:
                return f"{value:.4e}"
            return f"{value:.6f}"
        return str(value)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Format and log metrics at each logging step.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            logs: Metrics dict from the trainer
        """
        if logs is None or not state.is_world_process_zero:
            return

        # Build header: step + epoch
        parts = [f"step={state.global_step}"]
        if state.epoch is not None:
            parts.append(f"epoch={state.epoch:.2f}")

        reward_parts = []

        # Core keys first, in defined order
        for key in self._CORE_KEYS:
            if key in logs and isinstance(logs[key], (int, float)):
                parts.append(f"{key}={self._fmt(logs[key])}")

        # Aggregate reward (GRPO)
        for key in ("reward", "reward_std"):
            if key in logs and isinstance(logs[key], (int, float)):
                parts.append(f"{key}={self._fmt(logs[key])}")

        # Remaining metrics (sorted) and reward breakdown
        skip = set(self._CORE_KEYS) | {"reward", "reward_std", "total_flos", "epoch"}
        for key in sorted(logs):
            if key in skip or not isinstance(logs[key], (int, float)):
                continue
            if key.startswith(self._REWARD_PREFIX):
                short_key = key[len(self._REWARD_PREFIX):]
                reward_parts.append(f"{short_key}={self._fmt(logs[key])}")
            else:
                parts.append(f"{key}={self._fmt(logs[key])}")

        logger.info("  ".join(parts))

        # Log reward breakdown (GRPO only)
        if reward_parts:
            logger.info("  rewards: " + "  ".join(reward_parts))


class TeacherEvalCallback(TrainerCallback):
    """
    Evaluate co-trained teacher at each eval step.

    Runs the teacher through the same eval pipeline as the model and injects
    metrics into the metrics dict with an ``eval_teacher_`` prefix. This makes
    teacher metrics visible in W&B charts and to downstream callbacks like
    PeakMetricsCallback (add this callback *before* PeakMetricsCallback).
    """

    def __init__(self, trainer, teacher_label_map: Optional[List[int]] = None):
        """
        Initialize teacher evaluation callback.

        Args:
            trainer: DistillationTrainer instance (provides teacher, eval_dataset, etc.)
            teacher_label_map: Optional label index remapping for teacher logits
        """
        self.trainer = trainer
        self.teacher_label_map = teacher_label_map

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Run teacher evaluation and inject metrics after each model eval.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            metrics: Mutable metrics dict from model eval
        """
        if metrics is None:
            return

        from src.evaluation.evaluate import evaluate_model, _remap_compute_metrics

        compute_fn = self.trainer.compute_metrics
        if self.teacher_label_map is not None:
            compute_fn = _remap_compute_metrics(compute_fn, self.teacher_label_map)

        device = str(next(self.trainer.teacher.parameters()).device)

        logger.info("Evaluating teacher (mid-training)...")
        teacher_metrics = evaluate_model(
            model=self.trainer.teacher,
            eval_dataset=self.trainer.eval_dataset,
            tokenizer=self.trainer.processing_class,
            compute_metrics=compute_fn,
            data_collator=self.trainer.data_collator,
            batch_size=args.per_device_eval_batch_size,
            device=device,
            remove_unused_columns=args.remove_unused_columns,
        )

        # Inject into metrics dict so PeakMetricsCallback sees them
        for k, v in teacher_metrics.items():
            metrics[f"eval_teacher_{k}"] = v

        # HF Trainer calls self.log() BEFORE on_evaluate, so teacher metrics
        # miss the normal logging path. Re-invoke self.log() to push them
        # through the full pipeline (W&B, TensorBoard, console, etc.).
        self.trainer.log({f"eval_teacher_{k}": v for k, v in teacher_metrics.items()})

        logger.info(
            "  Teacher eval: "
            + ", ".join(f"{k}={v:.4f}" for k, v in teacher_metrics.items() if isinstance(v, float))
        )

    def on_save(self, args, state, control, **kwargs):
        """Save co-trained teacher alongside model at each checkpoint."""
        if self.trainer.freeze_teacher:
            return
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        teacher_dir = os.path.join(checkpoint_dir, "teacher")
        logger.info(f"Saving co-trained teacher to {teacher_dir}...")
        self.trainer.teacher.save_pretrained(teacher_dir)
        self.trainer.processing_class.save_pretrained(teacher_dir)


class TranslationMetricsCallback(TrainerCallback):
    """
    Compute translation metrics (BLEU, COMET) at end of each epoch.

    Generates translations on a subset of the evaluation dataset and computes
    BLEU and COMET scores. Model-agnostic - works with any seq2seq task.
    """

    def __init__(
        self,
        eval_dataset,
        tokenizer,
        source_lang: str = "en",
        target_lang: str = "fr",
        num_samples: int = 200,
        max_new_tokens: int = 128,
        batch_size: int = 8,
    ):
        """
        Initialize translation metrics callback.

        Args:
            eval_dataset: HuggingFace dataset with 'translation' field
            tokenizer: Model tokenizer
            source_lang: Source language code
            target_lang: Target language code
            num_samples: Number of examples to evaluate
            max_new_tokens: Maximum tokens to generate
            batch_size: Batch size for generation
        """
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.num_samples = min(num_samples, len(eval_dataset))
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

        # Load metrics
        try:
            import evaluate
            self.bleu_metric = evaluate.load("sacrebleu")
            self.comet_metric = evaluate.load("comet")
            logger.info(f"Translation metrics callback initialized ({self.num_samples} samples)")
        except ImportError as e:
            logger.warning(f"Failed to load metrics (missing dependency): {e}")
            self.bleu_metric = None
            self.comet_metric = None

    def on_epoch_end(self, args, state, control, model, **kwargs):
        """
        Compute translation metrics at end of epoch.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            model: Model being trained
            **kwargs: Additional arguments

        Returns:
            Updated control
        """
        if self.bleu_metric is None or self.comet_metric is None:
            return control

        logger.info("=" * 60)
        logger.info(f"Computing translation metrics (epoch {state.epoch})...")
        logger.info("=" * 60)

        # Get subset of eval dataset
        eval_subset = self.eval_dataset.select(range(self.num_samples))

        # Extract source texts and references
        sources = []
        references = []

        # Load raw dataset to get original text
        from datasets import load_dataset
        dataset = load_dataset("Helsinki-NLP/europarl", f"{self.source_lang}-{self.target_lang}", split="train")

        # Use same split logic as load_europarl
        splits = dataset.train_test_split(test_size=0.3, seed=42)
        temp_splits = splits["test"].train_test_split(test_size=0.333, seed=42)
        val_data = temp_splits["train"]

        for i in range(self.num_samples):
            example = val_data[i]
            sources.append(example["translation"][self.source_lang])
            references.append(example["translation"][self.target_lang])

        # Generate translations
        predictions = self._generate_translations(model, sources)

        # Compute metrics
        try:
            bleu_result = self.bleu_metric.compute(predictions=predictions, references=references)
            comet_result = self.comet_metric.compute(
                predictions=predictions,
                references=references,
                sources=sources
            )

            bleu_score = bleu_result["score"]
            comet_score = comet_result["mean_score"]

            logger.info(f"BLEU: {bleu_score:.2f}")
            logger.info(f"COMET: {comet_score:.4f}")

            # Log to W&B if available
            if args.report_to and "wandb" in args.report_to:
                try:
                    import wandb
                    wandb.log({
                        "eval/bleu": bleu_score,
                        "eval/comet": comet_score,
                        "epoch": state.epoch,
                    })
                except Exception as e:
                    logger.warning(f"Failed to log to W&B: {e}")

        except (ValueError, RuntimeError) as e:
            logger.error(f"Failed to compute metrics: {e}")

        logger.info("=" * 60)

        return control

    def _generate_translations(self, model, sources):
        """Generate translations for source texts.

        Args:
            model: Model to use for generation
            sources: List of source texts

        Returns:
            List of translated texts
        """
        from src.inference import InferenceEngine

        model.eval()
        engine = InferenceEngine.from_model(model, tokenizer=self.tokenizer)

        # Language names for prompt
        lang_names = {
            "en": "English", "fr": "French", "de": "German", "es": "Spanish",
            "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "da": "Danish",
        }
        source_lang_name = lang_names.get(self.source_lang, self.source_lang.upper())
        target_lang_name = lang_names.get(self.target_lang, self.target_lang.upper())

        system_message = f"You are a professional translator. Translate the provided text from {source_lang_name} to {target_lang_name}, remaining true to the source text. Do not add any additional commentary or conversational elements to your response."

        # Format prompts with chat template
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            prompts = []
            for source in sources:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": source}
                ]
                prompts.append(self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ))
        else:
            prompts = [f"{system_message}\n\n{source}\n\n" for source in sources]

        return engine.generate(
            prompts, max_new_tokens=self.max_new_tokens, batch_size=self.batch_size
        )


class PeakMetricsCallback(TrainerCallback):
    """
    Track and log peak metric values with checkpoint info.

    Records the best value achieved for each metric along with the
    step/epoch where it occurred. Logs to W&B at end of training.
    """

    def __init__(self, metrics_to_track: list[str] = None):
        """
        Initialize peak metrics tracker.

        Args:
            metrics_to_track: List of metric names to track (e.g., ["accuracy", "f1", "mcc"]).
                              If None, tracks all eval metrics.
        """
        self.metrics_to_track = metrics_to_track
        self.peaks = {}  # metric_name -> {"value": float, "step": int, "epoch": float}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Update peak values after each evaluation.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            metrics: Evaluation metrics dict
        """
        if metrics is None:
            return

        # Metrics that aren't model quality (runtime stats, not "peaks")
        skip_metrics = {"runtime", "samples_per_second", "steps_per_second", "model_preparation_time"}

        for key, value in metrics.items():
            # Skip non-eval metrics and loss (lower is better for loss)
            if not key.startswith("eval_") or "loss" in key:
                continue

            metric_name = key.replace("eval_", "")

            # Normalize teacher-prefixed metrics (e.g. "teacher_runtime" -> "runtime")
            base_name = metric_name.removeprefix("teacher_")

            # Skip runtime/throughput metrics
            if base_name in skip_metrics:
                continue

            # Filter to tracked metrics if specified
            if self.metrics_to_track and metric_name not in self.metrics_to_track:
                continue

            # Update peak if this is a new best (higher is better)
            if metric_name not in self.peaks or value > self.peaks[metric_name]["value"]:
                self.peaks[metric_name] = {
                    "value": value,
                    "step": state.global_step,
                    "epoch": state.epoch,
                }
                logger.info(f"New peak {metric_name}: {value:.4f} at step {state.global_step}")

                # Update W&B summary immediately
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.run.summary[f"peak/{metric_name}"] = value
                        wandb.run.summary[f"peak/{metric_name}_step"] = state.global_step
                        wandb.run.summary[f"peak/{metric_name}_epoch"] = state.epoch
                except ImportError:
                    pass

    def on_train_end(self, args, state, control, **kwargs):
        """
        Log peak metrics summary at end of training.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
        """
        if not self.peaks:
            return

        # Log summary to console
        logger.info("=" * 50)
        logger.info("PEAK METRICS SUMMARY")
        logger.info("=" * 50)
        for metric_name, peak_info in self.peaks.items():
            logger.info(
                f"  {metric_name}: {peak_info['value']:.4f} "
                f"(step {peak_info['step']}, epoch {peak_info['epoch']:.2f})"
            )
        logger.info("=" * 50)

    def get_peaks(self) -> dict:
        """Return the current peak metrics."""
        return self.peaks
