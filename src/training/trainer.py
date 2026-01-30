"""DistillationTrainer: HuggingFace Trainer subclass for knowledge distillation."""

from typing import Optional

import torch
from transformers import Trainer

from .losses import DistillationLossStrategy, KLDivergenceLoss, get_loss_strategy


class DistillationTrainer(Trainer):
    """
    Custom Trainer for knowledge distillation.

    Extends HuggingFace Trainer to:
    1. Maintain a teacher model (frozen, eval mode)
    2. Compute combined loss using pluggable loss strategies
    3. Log loss components separately for monitoring

    Uses Strategy Pattern for loss functions - swap loss strategies
    without modifying this class.
    """

    def __init__(
            self,
            teacher_model: torch.nn.Module,
            temperature: float = 2.0,
            alpha: float = 0.5,
            loss_strategy: Optional[DistillationLossStrategy] = None,
            loss_type: Optional[str] = None,
            *args,
            **kwargs
    ):
        """
        Initialize DistillationTrainer.

        Args:
            teacher_model: Pretrained teacher model (will be frozen)
            temperature: Temperature for softening distributions
            alpha: Weight for distillation loss (0=only CE, 1=only KL)
            loss_strategy: Pre-configured loss strategy instance (takes precedence)
            loss_type: Name of loss strategy to use (e.g., "kl_divergence", "mse_logits")
            *args: Additional arguments for Trainer
            **kwargs: Additional keyword arguments for Trainer
        """
        super().__init__(*args, **kwargs)

        self.teacher = teacher_model
        self.teacher.eval()  # Always in evaluation mode

        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False

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
            self.loss_strategy = KLDivergenceLoss(
                temperature=temperature,
                alpha=alpha
            )

        print(f"DistillationTrainer initialized:")
        print(f"  Loss strategy: {self.loss_strategy.name}")
        print(f"  Temperature: {temperature}")
        print(f"  Alpha: {alpha}")
        print(f"  Teacher frozen: {not any(p.requires_grad for p in self.teacher.parameters())}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute distillation loss using the configured strategy.

        Args:
            model: Student model
            inputs: Batch of inputs
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (for gradient accumulation)

        Returns:
            Loss tensor, or (loss, outputs) if return_outputs=True
        """
        # Get student predictions
        outputs = model(**inputs)
        student_logits = outputs.logits

        # Get teacher predictions (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.logits

        # Task loss (cross-entropy with labels)
        ce_loss = outputs.loss

        # Get labels if available
        labels = inputs.get("labels", None)

        # Compute loss using strategy
        loss_output = self.loss_strategy.compute(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            ce_loss=ce_loss,
            labels=labels
        )

        # Log individual components for monitoring
        log_dict = {"loss_total": loss_output.total_loss.item()}
        for name, value in loss_output.components.items():
            log_dict[name] = value.item() if torch.is_tensor(value) else value
        self.log(log_dict)

        return (loss_output.total_loss, outputs) if return_outputs else loss_output.total_loss
