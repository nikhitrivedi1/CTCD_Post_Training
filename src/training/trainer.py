"""Forge trainers: FinetuningTrainer (base) and DistillationTrainer (KD)."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import Trainer

from src.utils.config import resolve_distillation_coefficients

from .losses import (
    DistillationLossStrategy,
    KLDistillationLoss,
    build_distill_topology_strategy,
    build_layer_mapping_and_projector,
    build_teacher_response_only_strategy,
    compute_entropy_of_output,
    get_loss_strategy,
)

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

    Supports three modes:
    - Traditional distillation: Teacher is frozen, provides fixed soft labels (beta_h = 0, beta_s = 0)
    - Co-training: Teacher trains alongside model, soft labels improve over time (beta_h = 1, beta_s = 0)
    - Co-training with codistillation: Teacher trains alongside model, soft labels improve over time (beta_h = 1, beta_s = 1)

    Uses Strategy Pattern for loss functions - swap loss strategies without modifying this class.
    """

    def __init__(
            self,
            teacher_model: torch.nn.Module,
            temperature: float = 2.0,
            alpha_h: float = 1.0,
            alpha_s: float = 1.0,
            beta_h: Optional[float] = None,
            beta_s: Optional[float] = None,
            freeze_teacher: bool = True,
            loss_strategy: Optional[DistillationLossStrategy] = None,
            teacher_loss_strategy: Optional[DistillationLossStrategy] = None,
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
            alpha_h: Hard (CE) weight for the student (default 0.5).
            alpha_s: Soft (distill) weight for the student (default 0.5).
            beta_h, beta_s: Hard / soft weights for the teacher when co-training (default 1, 0).
            freeze_teacher: If True, freeze teacher (traditional distillation).
                           If False, train teacher alongside model (co-training).
            loss_strategy: Pre-configured loss strategy instance (takes precedence)
            teacher_loss_strategy: Co-training only: response strategy with β baked in; if omitted,
                defaults to KL with ``ce_loss_coeff``/``kd_loss_coeff`` = ``beta_h``/``beta_s``.
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

        # Overall Param Init
        self.teacher = teacher_model
        self.freeze_teacher = freeze_teacher
        self.teacher_label_map = teacher_label_map
        self.temperature = temperature



        # Accumulate per-step metrics for averaged logging at logging_steps intervals.
        # Without this, self.log() fires every step, creating noisy W&B charts.
        self._metric_accumulator = {}
        self._metric_count = 0

        # Filled in on_pre_optimizer_step (co-training) for the next HF training log bundle.
        self._last_student_teacher_grad_norms: Optional[Tuple[float, float]] = None

        # ===========================================================================
        #  Student Model Initialization
        # ===========================================================================
        self.alpha_h = float(alpha_h)
        self.alpha_s = float(alpha_s)

        # Set up loss strategy (Strategy Pattern) - for the student
        if loss_strategy is not None:
            # Use provided strategy instance
            self.loss_strategy = loss_strategy
        elif loss_type is not None:
            # Create strategy from name
            self.loss_strategy = get_loss_strategy(
                loss_type,
                temperature=temperature,
                alpha_h=self.alpha_h,
                alpha_s=self.alpha_s,
            )
        else:
            # Default to KL divergence
            self.loss_strategy = KLDistillationLoss(
                temperature=temperature,
                ce_loss_coeff=self.alpha_h,
                kd_loss_coeff=self.alpha_s,
            )

        # Store layer mapping and projector for feature/attention distillation
        self.layer_mapping = layer_mapping
        self.layer_mapping_type = layer_mapping_type
        self.projector = projector

        # Track what outputs we need based on loss strategy requirements
        self.output_hidden_states = getattr(self.loss_strategy, 'requires_hidden_states', False)
        self.output_attentions = getattr(self.loss_strategy, 'requires_attentions', False)
        
        # ===========================================================================
        #  Teacher Model Initialization
        # ===========================================================================
        if freeze_teacher:
            self.teacher.eval()  # Evaluation mode (no dropout)
            for param in self.teacher.parameters():
                param.requires_grad = False
        else:
            self.teacher.train()  # Training mode for co-training


        if freeze_teacher:
            self.beta_h = 0.0
            self.beta_s = 0.0
        elif beta_h is None and beta_s is None:
            self.beta_h, self.beta_s = 1.0, 0.0
        else:
            self.beta_s = float(beta_s)
            self.beta_h = float(beta_h)


        if freeze_teacher:
            self.teacher_loss_strategy = None
        elif teacher_loss_strategy is not None:
            self.teacher_loss_strategy = teacher_loss_strategy
        else:
            self.teacher_loss_strategy = KLDistillationLoss(
                temperature=temperature,
                ce_loss_coeff=self.beta_h,
                kd_loss_coeff=self.beta_s,
            )



        logger.info("DistillationTrainer initialized:")
        logger.info(f"  Loss strategy: {self.loss_strategy.name}")
        logger.info(f"  Temperature: {temperature}")
        logger.info(f"  alpha_h: {self.alpha_h}, alpha_s: {self.alpha_s}")
        if not freeze_teacher:
            logger.info(f"  beta_h: {self.beta_h}, beta_s: {self.beta_s}")
        logger.info(f"  Teacher frozen: {freeze_teacher}")
        if not freeze_teacher and self.teacher_loss_strategy is not None:
            logger.info(f"  Teacher loss strategy: {self.teacher_loss_strategy.name}")
        if self.output_hidden_states:
            logger.info(f"  Output hidden states: True (required by {self.loss_strategy.name})")
        if self.output_attentions:
            logger.info(f"  Output attentions: True (required by {self.loss_strategy.name})")

    def _record_student_teacher_grad_norms_for_logging(self) -> None:
        """
        Before the optimizer step, store L2 grad norms for student vs teacher.
        Called from a callback at ``on_pre_optimizer_step`` (before HF Trainer's
        own clip_grad_norm_ call); values are merged into the next training ``log()``.

        Does not call accelerator.unscale_gradients() — HF Trainer calls it
        immediately after on_pre_optimizer_step for gradient clipping, and calling
        unscale_() twice on the same optimizer raises a RuntimeError. Instead, raw
        norms are read from scaled gradients and divided by the AMP scale factor to
        recover true gradient magnitudes.
        """
        if self.freeze_teacher:
            return
        student_mod = self.accelerator.unwrap_model(self.model)
        student_params = [p for p in student_mod.parameters() if p.requires_grad]
        teacher_params = [p for p in self.teacher.parameters() if p.requires_grad]

        scale = self.accelerator.scaler.get_scale() if self.accelerator.scaler is not None else 1.0

        if not student_params:
            student_norm = 0.0
        else:
            sn = torch.nn.utils.clip_grad_norm_(student_params, float("inf"))
            student_norm = float(sn.item() if torch.is_tensor(sn) else sn) / scale

        if not teacher_params:
            teacher_norm = 0.0
        else:
            tn = torch.nn.utils.clip_grad_norm_(teacher_params, float("inf"))
            teacher_norm = float(tn.item() if torch.is_tensor(tn) else tn) / scale

        self._last_student_teacher_grad_norms = (student_norm, teacher_norm)

    def log(self, logs, start_time=None):
        """
        Attach student/teacher grad norms to the standard HF training log dict (the one
        that includes ``loss``), without mixing them into other ``log()`` calls.
        """
        if (
            isinstance(logs, dict)
            and not self.freeze_teacher
            and "loss" in logs
            and self._last_student_teacher_grad_norms is not None
        ):
            student_gn, teacher_gn = self._last_student_teacher_grad_norms
            logs["student_grad_norm"] = student_gn
            logs["teacher_grad_norm"] = teacher_gn
            self._last_student_teacher_grad_norms = None
        return super().log(logs, start_time=start_time)

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

    def _detach_output_intermediates(self, outputs):
        """
        Detach hidden states and attentions from model outputs (logits stay connected).

        Used for the student branch in co-training so the teacher does not receive gradients
        through student-side F/A distillation; v1 teacher CTCD uses detached student logits only.

        Args:
            outputs: Model outputs (e.g. teacher forward) with optional hidden_states/attentions

        Returns:
            Shallow copy with detached intermediates when present
        """
        # Create a shallow copy to avoid modifying the original
        # (we still need original .loss for teacher CE computation)
        from dataclasses import is_dataclass

        if is_dataclass(outputs):
            updates = {}
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                updates["hidden_states"] = tuple(h.detach() for h in outputs.hidden_states)
            if hasattr(outputs, "attentions") and outputs.attentions is not None:
                updates["attentions"] = tuple(a.detach() for a in outputs.attentions)

            if updates:
                from dataclasses import replace
                return replace(outputs, **updates)

        return outputs

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

        # Always-on entropy metrics for monitoring teacher vs student confidence.
        with torch.no_grad():
            teacher_entropy = compute_entropy_of_output(teacher_logits_for_distill)
            student_entropy = compute_entropy_of_output(model_logits.detach())

        # Also detach hidden states and attentions for Feature/Attention losses
        # This prevents the bug where teacher gets pulled toward model representations
        teacher_outputs_for_distill = teacher_outputs
        if not self.freeze_teacher:
            # Student branch: detach teacher F/A so gradients do not flow into teacher via KD/F/A.
            teacher_outputs_for_distill = self._detach_output_intermediates(teacher_outputs)

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

        # Student branch: CE + KD vs detached teacher (implemented by loss_strategy.compute)
        student_loss_output = self.loss_strategy.compute(
            model_logits=model_logits,
            teacher_logits=teacher_logits_for_distill,
            ce_loss=ce_loss,
            labels=labels,
            model_outputs=outputs,
            teacher_outputs=teacher_outputs_for_distill,
            **loss_kwargs
        )

        total_loss = student_loss_output.total_loss
        teacher_loss_output = None
        if not self.freeze_teacher:
            align_vocab = model_logits.size(-1)
            teacher_logits_aligned = teacher_logits[..., :align_vocab]
            # When teacher_label_map is set, teacher_outputs.loss was computed with
            # misaligned labels (HuggingFace computes loss before we remap logits).
            if self.teacher_label_map is not None and labels is not None:
                teacher_ce_loss = F.cross_entropy(teacher_logits_aligned, labels)
            else:
                teacher_ce_loss = teacher_outputs.loss

            teacher_loss_kwargs = {
                "training_progress": loss_kwargs["training_progress"],
            }
            # Invariant: __init__ always sets a strategy when co-training (explicit or KL fallback).
            tls = self.teacher_loss_strategy
            assert tls is not None
            teacher_loss_output = tls.compute(
                model_logits=teacher_logits_aligned,
                teacher_logits=model_logits.detach(),
                ce_loss=teacher_ce_loss,
                labels=labels,
                model_outputs=None,
                teacher_outputs=None,
                **teacher_loss_kwargs,
            )
            total_loss = total_loss + teacher_loss_output.total_loss

        # Accumulate loss components and log averages at logging_steps intervals.
        step_dict = {
            "total_loss": total_loss.item(),
            "teacher_entropy": teacher_entropy.item(),
            "student_entropy": student_entropy.item(),
        }
        for name, value in student_loss_output.components.items():
            step_dict[name] = value.item() if torch.is_tensor(value) else value
        if teacher_loss_output is not None:
            for name, value in teacher_loss_output.components.items():
                key = f"teacher_{name}"
                step_dict[key] = value.item() if torch.is_tensor(value) else value

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
        freeze_teacher = config.get("model", {}).get("teacher", {}).get("freeze", True)
        coeffs = resolve_distillation_coefficients(distill_config, freeze_teacher=freeze_teacher)

        # Basic distillation parameters
        trainer_kwargs = {
            "temperature": distill_config.get("temperature", 2.0),
            "alpha_h": coeffs["alpha_h"],
            "alpha_s": coeffs["alpha_s"],
            "beta_h": coeffs["beta_h"],
            "beta_s": coeffs["beta_s"],
        }

        temp = trainer_kwargs["temperature"]
        loss_strategy = build_distill_topology_strategy(
            distill_config, coeffs["alpha_h"], coeffs["alpha_s"], temperature=temp,
        )
        trainer_kwargs["loss_strategy"] = loss_strategy
        if not freeze_teacher:
            trainer_kwargs["teacher_loss_strategy"] = build_teacher_response_only_strategy(
                distill_config, coeffs["beta_h"], coeffs["beta_s"], temperature=temp,
            )

        # Teacher label remapping
        if "teacher_label_map" in distill_config:
            trainer_kwargs["teacher_label_map"] = distill_config["teacher_label_map"]

        device = str(next(model.parameters()).device)
        layer_mapping, projector = build_layer_mapping_and_projector(
            distill_config, loss_strategy, model, teacher_model, device,
        )
        layer_mapping_config = distill_config.get("layer_mapping", {})
        if layer_mapping_config:
            trainer_kwargs["layer_mapping_type"] = layer_mapping_config.get("type", "uniform")

        if projector is not None:
            p = next(model.parameters())
            projector = projector.to(device=p.device, dtype=p.dtype)

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
