"""DistillationTrainer: HuggingFace Trainer subclass for knowledge distillation."""

import torch
import torch.nn.functional as F
from transformers import Trainer


class DistillationTrainer(Trainer):
    """
    Custom Trainer for knowledge distillation.

    Extends HuggingFace Trainer to:
    1. Maintain a teacher model (frozen, eval mode)
    2. Compute combined loss (KL divergence + cross-entropy)
    3. Log both loss components separately for monitoring

    Based on the "Less is More" paper implementation.
    """

    def __init__(
            self,
            teacher_model: torch.nn.Module,
            temperature: float,
            alpha: float,
            *args,
            **kwargs
    ):
        """
        Initialize DistillationTrainer.

        Args:
            teacher_model: Pretrained teacher model (will be frozen)
            temperature: Temperature for softening distributions
            alpha: Weight for distillation loss (0=only CE, 1=only KL)
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

        print(f"DistillationTrainer initialized:")
        print(f"  Temperature: {temperature}")
        print(f"  Alpha: {alpha}")
        print(f"  Teacher frozen: {not any(p.requires_grad for p in self.teacher.parameters())}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute distillation loss combining KL divergence and cross-entropy.

        This is the core of knowledge distillation. We:
        1. Get student predictions (with gradients)
        2. Get teacher predictions (no gradients, frozen)
        3. Compute task loss (cross-entropy with true labels)
        4. Compute distillation loss (KL divergence with teacher)
        5. Combine with alpha weighting

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
        # This is computed internally by the model when labels are provided
        ce_loss = outputs.loss

        # Distillation loss (KL divergence)
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Combined loss
        loss = (1 - self.alpha) * ce_loss + self.alpha * kl_loss

        # Log individual components for monitoring
        self.log({
            "loss_ce": ce_loss.item(),
            "loss_kl": kl_loss.item(),
            "loss_total": loss.item(),
        })

        return (loss, outputs) if return_outputs else loss
