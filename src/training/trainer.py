"""Forge trainers: FinetuningTrainer (base) and DistillationTrainer (KD)."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import Trainer

from .layer_mapping import get_layer_mapping
from .losses import DistillationLossStrategy, KLDistillationLoss, get_loss_strategy
from .projections import create_projection_if_needed

logger = logging.getLogger(__name__)


class FinetuningTrainer(Trainer):
    """
    Base trainer for supervised fine-tuning.

    Wraps HuggingFace Trainer and serves as the base class for
    DistillationTrainer. Currently a pass-through that establishes
    the class hierarchy for future extensions (LoRA setup, eval hooks).
    """

    pass


class DistillationTrainer(FinetuningTrainer):
    """
    Custom Trainer for knowledge distillation.

    Extends HuggingFace Trainer to:
    1. Maintain a teacher model (frozen or co-trained)
    2. Compute combined loss using pluggable loss strategies
    3. Log loss components separately for monitoring

    Supports two modes:
    - Traditional distillation: Teacher is frozen, provides fixed soft labels
    - Co-training: Teacher trains alongside model, soft labels improve over time

    Uses Strategy Pattern for loss functions - swap loss strategies
    without modifying this class.
    """

    def __init__(
            self,
            teacher_model: torch.nn.Module,
            temperature: float = 2.0,
            alpha: float = 0.5,
            freeze_teacher: bool = True,
            loss_strategy: Optional[DistillationLossStrategy] = None,
            loss_type: Optional[str] = None,
            teacher_label_map: Optional[list] = None,
            layer_mapping_type: Optional[str] = None,
            layer_mapping: Optional[List[Tuple[int, int]]] = None,
            projector: Optional[torch.nn.Module] = None,
            *args,
            **kwargs
    ):
        """
        Initialize DistillationTrainer.

        Args:
            teacher_model: Pretrained teacher model
            temperature: Temperature for softening distributions
            alpha: Weight for distillation loss (0=only CE, 1=only KL)
            freeze_teacher: If True, freeze teacher (traditional distillation).
                           If False, train teacher alongside model (co-training).
            loss_strategy: Pre-configured loss strategy instance (takes precedence)
            loss_type: Name of loss strategy to use (e.g., "kl_divergence", "mse_logits")
            teacher_label_map: Optional list to remap teacher logits to match dataset labels.
                              e.g., [1, 2, 0] means teacher label 1 -> dataset label 0, etc.
                              Use when teacher was trained with different label ordering.
            layer_mapping_type: Strategy for mapping teacher layers to model layers
                               (uniform, last_n, first_last). Used for feature/attention distillation.
            layer_mapping: Explicit list of (teacher_idx, model_idx) pairs.
                          Overrides layer_mapping_type if provided.
            projector: Optional projection module for dimension alignment in feature distillation.
            *args: Additional arguments for Trainer
            **kwargs: Additional keyword arguments for Trainer
        """
        super().__init__(*args, **kwargs)

        self.teacher = teacher_model
        self.freeze_teacher = freeze_teacher
        self.teacher_label_map = teacher_label_map

        if freeze_teacher:
            self.teacher.eval()  # Evaluation mode (no dropout)
            for param in self.teacher.parameters():
                param.requires_grad = False
        else:
            self.teacher.train()  # Training mode for co-training

        self.temperature = temperature
        self.alpha = alpha

        # Set up loss strategy (Strategy Pattern)
        if loss_strategy is not None:
            # Use provided strategy instance
            self.loss_strategy = loss_strategy
        elif loss_type is not None:
            # Create strategy from name
            self.loss_strategy = get_loss_strategy(
                loss_type,
                temperature=temperature,
                alpha=alpha
            )
        else:
            # Default to KL divergence
            self.loss_strategy = KLDistillationLoss(
                temperature=temperature,
                alpha=alpha
            )

        # Store layer mapping and projector for feature/attention distillation
        self.layer_mapping = layer_mapping
        self.layer_mapping_type = layer_mapping_type
        self.projector = projector

        # Track what outputs we need based on loss strategy requirements
        self.output_hidden_states = getattr(self.loss_strategy, 'requires_hidden_states', False)
        self.output_attentions = getattr(self.loss_strategy, 'requires_attentions', False)

        # Accumulate per-step metrics for averaged logging at logging_steps intervals.
        # Without this, self.log() fires every step, creating noisy W&B charts.
        self._metric_accumulator = {}
        self._metric_count = 0

        logger.info("DistillationTrainer initialized:")
        logger.info(f"  Loss strategy: {self.loss_strategy.name}")
        logger.info(f"  Temperature: {temperature}")
        logger.info(f"  Alpha: {alpha}")
        logger.info(f"  Teacher frozen: {freeze_teacher}")
        if self.output_hidden_states:
            logger.info(f"  Output hidden states: True (required by {self.loss_strategy.name})")
        if self.output_attentions:
            logger.info(f"  Output attentions: True (required by {self.loss_strategy.name})")

    def create_optimizer(self):
        """
        Create optimizer, including teacher parameters if co-training.
        """
        if self.optimizer is not None:
            return self.optimizer

        # Get default optimizer for model
        super().create_optimizer()

        # If co-training, add teacher parameters to optimizer
        if not self.freeze_teacher:
            teacher_params = list(self.teacher.parameters())
            # Add teacher params to existing optimizer's param groups
            self.optimizer.add_param_group({"params": teacher_params})
            logger.info(f"  Added {len(teacher_params)} teacher parameter groups to optimizer")

        return self.optimizer

    def _detach_teacher_intermediates(self, teacher_outputs):
        """
        Detach hidden states and attentions from teacher outputs.

        In co-training, teacher should only learn from CE loss, not from
        distillation losses (F, A). This prevents the teacher from being
        pulled toward matching the (worse) model's representations.

        Args:
            teacher_outputs: Model outputs containing hidden_states/attentions

        Returns:
            Modified outputs with detached intermediate representations
        """
        # Create a shallow copy to avoid modifying the original
        # (we still need original .loss for teacher CE computation)
        from dataclasses import is_dataclass

        if is_dataclass(teacher_outputs):
            # HuggingFace model outputs are dataclasses
            updates = {}
            if hasattr(teacher_outputs, "hidden_states") and teacher_outputs.hidden_states is not None:
                updates["hidden_states"] = tuple(h.detach() for h in teacher_outputs.hidden_states)
            if hasattr(teacher_outputs, "attentions") and teacher_outputs.attentions is not None:
                updates["attentions"] = tuple(a.detach() for a in teacher_outputs.attentions)

            if updates:
                # Use replace() to create a new instance with updated fields
                from dataclasses import replace
                return replace(teacher_outputs, **updates)

        return teacher_outputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute distillation loss using the configured strategy.

        Args:
            model: Model being trained
            inputs: Batch of inputs
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (for gradient accumulation)

        Returns:
            Loss tensor, or (loss, outputs) if return_outputs=True
        """
        # Build extra kwargs for model forward pass if needed
        model_kwargs = {}
        if self.output_hidden_states:
            model_kwargs["output_hidden_states"] = True
        if self.output_attentions:
            model_kwargs["output_attentions"] = True

        # Get model predictions
        outputs = model(**inputs, **model_kwargs)

        # During evaluation, just return CE loss (skip distillation loss computation)
        # Distillation loss requires hidden_states/attentions which are disabled during
        # eval to avoid OOM from accumulating large intermediate tensors
        if not model.training:
            return (outputs.loss, outputs) if return_outputs else outputs.loss
        model_logits = outputs.logits

        # Get teacher predictions
        if self.freeze_teacher:
            # Traditional distillation: no gradients through teacher
            with torch.no_grad():
                teacher_outputs = self.teacher(**inputs, **model_kwargs)
                teacher_logits = teacher_outputs.logits
        else:
            # Co-training: gradients flow through teacher
            teacher_outputs = self.teacher(**inputs, **model_kwargs)
            teacher_logits = teacher_outputs.logits

        # Remap teacher logits if label ordering differs from dataset
        # e.g., TextAttack MNLI uses [contradiction, entailment, neutral]
        # but HuggingFace MNLI uses [entailment, neutral, contradiction]
        if self.teacher_label_map is not None:
            teacher_logits = teacher_logits[:, self.teacher_label_map]

        # Detach teacher outputs for distillation loss computation
        # Teacher should not receive gradients through distillation losses (KL, F, A)
        # - only through its own CE loss (computed separately below)
        teacher_logits_for_distill = teacher_logits.detach()

        # Align vocab dimensions for cross-tokenizer distillation
        # (e.g., Qwen3-VL has 4 extra special tokens vs Qwen2.5-VL)
        if model_logits.size(-1) != teacher_logits_for_distill.size(-1):
            min_vocab = min(model_logits.size(-1), teacher_logits_for_distill.size(-1))
            model_logits = model_logits[..., :min_vocab]
            teacher_logits_for_distill = teacher_logits_for_distill[..., :min_vocab]

        # Also detach hidden states and attentions for Feature/Attention losses
        # This prevents the bug where teacher gets pulled toward model representations
        teacher_outputs_for_distill = teacher_outputs
        if not self.freeze_teacher:
            # In co-training, we need to detach intermediate outputs
            # (when frozen, they're already detached via torch.no_grad context)
            teacher_outputs_for_distill = self._detach_teacher_intermediates(teacher_outputs)

        # Task loss (cross-entropy with labels)
        ce_loss = outputs.loss

        # Get labels if available
        labels = inputs.get("labels", None)

        # Build kwargs for loss strategy
        loss_kwargs = {}
        if self.layer_mapping is not None:
            loss_kwargs["layer_mapping"] = self.layer_mapping
        if self.projector is not None:
            loss_kwargs["projector"] = self.projector

        # Pass training progress for staged distillation
        if hasattr(self, "state") and self.state.max_steps > 0:
            loss_kwargs["training_progress"] = self.state.global_step / self.state.max_steps
        else:
            loss_kwargs["training_progress"] = 0.0

        # Compute loss using strategy (with detached teacher outputs)
        loss_output = self.loss_strategy.compute(
            model_logits=model_logits,
            teacher_logits=teacher_logits_for_distill,
            ce_loss=ce_loss,
            labels=labels,
            model_outputs=outputs,
            teacher_outputs=teacher_outputs_for_distill,
            **loss_kwargs
        )

        # Combine losses: model loss + teacher CE (if co-training)
        total_loss = loss_output.total_loss
        if not self.freeze_teacher:
            # When teacher_label_map is set, teacher_outputs.loss was computed with
            # misaligned labels (HuggingFace computes loss before we remap logits).
            # Recompute teacher CE using remapped logits for correct gradients.
            if self.teacher_label_map is not None and labels is not None:
                # teacher_logits already remapped at line 188
                teacher_ce_loss = F.cross_entropy(teacher_logits, labels)
            else:
                teacher_ce_loss = teacher_outputs.loss
            total_loss = total_loss + teacher_ce_loss

        # Accumulate loss components and log averages at logging_steps intervals.
        step_dict = {"total_loss": total_loss.item()}
        for name, value in loss_output.components.items():
            step_dict[name] = value.item() if torch.is_tensor(value) else value
        if not self.freeze_teacher:
            step_dict["teacher_ce_loss"] = teacher_ce_loss.item()

        for key, value in step_dict.items():
            self._metric_accumulator[key] = self._metric_accumulator.get(key, 0.0) + value
        self._metric_count += 1

        # Flush averaged metrics at logging intervals
        if (hasattr(self, 'args') and self.args.logging_steps > 0
                and self._metric_count >= self.args.logging_steps):
            avg_dict = {k: v / self._metric_count for k, v in self._metric_accumulator.items()}
            self.log(avg_dict)
            self._metric_accumulator = {}
            self._metric_count = 0

        return (total_loss, outputs) if return_outputs else total_loss

    @classmethod
    def setup_from_config(
        cls,
        config: Dict[str, Any],
        teacher_model: torch.nn.Module,
        model: torch.nn.Module,
    ) -> Tuple[Dict[str, Any], Optional[List[Tuple[int, int]]], Optional[torch.nn.Module]]:
        """
        Parse distillation config and prepare trainer arguments.

        This factory method extracts distillation-related configuration and
        sets up layer mapping and projections for feature/attention distillation.

        Args:
            config: Full configuration dictionary (with 'distillation' key)
            teacher_model: Teacher model (for layer/dimension info)
            model: Model (for layer/dimension info)

        Returns:
            Tuple of (trainer_kwargs, layer_mapping, projector)
            - trainer_kwargs: Dict of arguments to pass to DistillationTrainer
            - layer_mapping: List of (teacher_idx, model_idx) pairs or None
            - projector: Projection module or None

        Example:
            trainer_kwargs, layer_mapping, projector = DistillationTrainer.setup_from_config(
                config, teacher, model
            )
            trainer = DistillationTrainer(
                teacher_model=teacher,
                model=model,
                **trainer_kwargs,
                layer_mapping=layer_mapping,
                projector=projector,
                ...
            )
        """
        distill_config = config.get("distillation", {})

        # Basic distillation parameters
        trainer_kwargs = {
            "temperature": distill_config.get("temperature", 2.0),
            "alpha": distill_config.get("alpha", 0.5),
        }

        # Handle staged distillation or simple loss type
        if "stages" in distill_config:
            # Staged distillation: build StagedDistillationLoss
            from .losses import StagedDistillationLoss
            trainer_kwargs["loss_strategy"] = StagedDistillationLoss(
                stages=distill_config["stages"],
                alpha=trainer_kwargs["alpha"],
                temperature=trainer_kwargs["temperature"],
            )
        else:
            trainer_kwargs["loss_type"] = distill_config.get("loss_type", "kl_divergence")

        # Teacher label remapping
        if "teacher_label_map" in distill_config:
            trainer_kwargs["teacher_label_map"] = distill_config["teacher_label_map"]

        # Layer mapping for feature/attention distillation
        layer_mapping = None
        layer_mapping_config = distill_config.get("layer_mapping", {})
        if layer_mapping_config:
            mapping_type = layer_mapping_config.get("type", "uniform")

            # Get layer counts from model configs
            teacher_layers = getattr(teacher_model.config, "num_hidden_layers", 12)
            model_layers = getattr(model.config, "num_hidden_layers", 6)

            if mapping_type == "custom" and "mapping" in layer_mapping_config:
                # Explicit mapping provided
                layer_mapping = [
                    (pair["teacher"], pair["model"])
                    for pair in layer_mapping_config["mapping"]
                ]
            else:
                # Use a mapping strategy
                mapper = get_layer_mapping(mapping_type)
                layer_mapping = mapper.map(teacher_layers, model_layers)

            trainer_kwargs["layer_mapping_type"] = mapping_type

        # Projection for dimension alignment
        projector = None
        projection_config = distill_config.get("projection", {})
        if projection_config:
            teacher_dim = getattr(teacher_model.config, "hidden_size", 768)
            model_dim = getattr(model.config, "hidden_size", 768)

            if teacher_dim != model_dim or projection_config.get("force", False):
                proj_type = projection_config.get("type", "linear")
                projector = create_projection_if_needed(
                    teacher_dim=teacher_dim,
                    model_dim=model_dim,
                    projection_type=proj_type,
                    hidden_dim=projection_config.get("hidden_dim"),
                    activation=projection_config.get("activation", "relu"),
                )
                # Move projector to same device and dtype as model
                if hasattr(model, "device"):
                    projector = projector.to(device=model.device, dtype=model.dtype)

        return trainer_kwargs, layer_mapping, projector

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """
        Override prediction_step to disable hidden states/attentions during eval.

        During training, we need hidden_states and attentions for distillation loss.
        During evaluation, we only need logits for metrics - outputting hidden states
        and attentions causes OOM when accumulated across the full eval set.

        We temporarily disable self.output_hidden_states and self.output_attentions
        so that compute_loss (if called) won't add these kwargs to the forward pass.
        """
        # Temporarily disable hidden states/attentions for eval
        old_hidden = self.output_hidden_states
        old_attn = self.output_attentions
        self.output_hidden_states = False
        self.output_attentions = False

        try:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
            )
        finally:
            # Restore original settings
            self.output_hidden_states = old_hidden
            self.output_attentions = old_attn
