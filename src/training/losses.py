"""Loss functions for knowledge distillation."""

import torch
import torch.nn.functional as F


def kl_divergence_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float):
    """
    Compute KL divergence loss for knowledge distillation.

    The KL divergence measures how different the student's output distribution
    is from the teacher's. We use temperature scaling to soften the distributions,
    making it easier for the student to learn.

    Args:
        student_logits: Raw output logits from student model
        teacher_logits: Raw output logits from teacher model
        temperature: Temperature for softening probability distributions

    Returns:
        KL divergence loss (scalar tensor)
    """
    # Apply temperature scaling and compute softmax
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # Compute KL divergence
    # Note: F.kl_div expects log probabilities for the first argument
    kl_loss = F.kl_div(
        student_probs,
        teacher_probs,
        reduction='batchmean'
    )

    # Scale by temperature^2 to correct for temperature scaling
    # This is a standard practice in distillation (Hinton et al., 2015)
    return kl_loss * (temperature ** 2)


def distillation_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        ce_loss: torch.Tensor,
        temperature: float,
        alpha: float
):
    """
    Compute combined distillation loss.

    Combines the cross-entropy loss (task-specific) with KL divergence loss
    (knowledge from teacher). The alpha parameter controls the balance.

    Loss = (1 - alpha) * CE_loss + alpha * KL_loss

    Args:
        student_logits: Raw output logits from student model
        teacher_logits: Raw output logits from teacher model
        labels: Ground truth labels
        ce_loss: Pre-computed cross-entropy loss
        temperature: Temperature for distillation
        alpha: Weight for distillation loss (0 = only CE, 1 = only KL)

    Returns:
        Combined loss (scalar tensor)
    """
    # Compute KL divergence loss
    kl_loss = kl_divergence_loss(student_logits, teacher_logits, temperature)

    # Combine losses
    total_loss = (1 - alpha) * ce_loss + alpha * kl_loss

    return total_loss
