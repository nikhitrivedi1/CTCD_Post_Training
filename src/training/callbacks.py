"""Callbacks for SLURM signal handling and translation metrics evaluation."""

import logging
import signal
from typing import Optional

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


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

        logger.info("SLURM signal handler registered (SIGUSR1, SIGTERM)")

    def _handler(self, signum, frame):
        """
        Signal handler that triggers checkpoint saving.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        self.caught_signal = True
        logger.warning("=" * 60)
        logger.warning(f"Caught signal {signum} - saving checkpoint...")
        logger.warning("=" * 60)

        # Save model and trainer state
        self.trainer.save_model()
        self.trainer.save_state()

        logger.info("Checkpoint saved successfully")

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
        """
        Generate translations for source texts.

        Args:
            model: Model to use for generation
            sources: List of source texts

        Returns:
            List of translated texts
        """
        model.eval()
        device = next(model.parameters()).device
        predictions = []

        # Language names for prompt
        lang_names = {
            "en": "English", "fr": "French", "de": "German", "es": "Spanish",
            "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "da": "Danish",
        }
        source_lang_name = lang_names.get(self.source_lang, self.source_lang.upper())
        target_lang_name = lang_names.get(self.target_lang, self.target_lang.upper())

        system_message = f"You are a professional translator. Translate the provided text from {source_lang_name} to {target_lang_name}, remaining true to the source text. Do not add any additional commentary or conversational elements to your response."

        for i in range(0, len(sources), self.batch_size):
            batch_sources = sources[i:i + self.batch_size]

            # Format with chat template if available
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                prompts = []
                for source in batch_sources:
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": source}
                    ]
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    prompts.append(prompt)
            else:
                prompts = [f"{system_message}\n\n{source}\n\n" for source in batch_sources]

            # Tokenize
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode and extract translation
            for output in outputs:
                full_text = self.tokenizer.decode(output, skip_special_tokens=False)

                # Extract response after assistant header
                if "<|end_header_id|>" in full_text:
                    parts = full_text.split("<|end_header_id|>")
                    if len(parts) > 3:
                        translation = parts[3].split("<|eot_id|>")[0].strip()
                    else:
                        translation = ""
                else:
                    # Fallback: take text after prompt
                    translation = self.tokenizer.decode(output, skip_special_tokens=True)
                    # Remove prompt if still present
                    for prompt in prompts:
                        if translation.startswith(prompt):
                            translation = translation[len(prompt):].strip()
                            break

                predictions.append(translation)

        return predictions


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
        skip_metrics = {"runtime", "samples_per_second", "steps_per_second"}

        for key, value in metrics.items():
            # Skip non-eval metrics and loss (lower is better for loss)
            if not key.startswith("eval_") or "loss" in key:
                continue

            metric_name = key.replace("eval_", "")

            # Skip runtime/throughput metrics
            if metric_name in skip_metrics:
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
