import torch
import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, y_true, alpha=0.5, T=2.0):
    """
    Combined hard-label + soft-target knowledge distillation loss.

    Args:
        student_logits: Tensor (batch, 1) or (batch,)
        teacher_logits: Tensor (batch, 1) or (batch,)
        y_true: Tensor (batch,) float binary labels
        alpha: weight for soft distillation loss
        T: distillation temperature

    Returns:
        scalar loss
    """
    hard_loss = F.binary_cross_entropy_with_logits(
        student_logits.view(-1), y_true.float()
    )

    # Convert binary logits to 2-class soft distributions for KL divergence
    s2 = torch.cat([-student_logits.view(-1, 1) / T,
                     student_logits.view(-1, 1) / T], dim=1)
    t2 = torch.cat([-teacher_logits.view(-1, 1) / T,
                     teacher_logits.view(-1, 1) / T], dim=1)

    soft_loss = F.kl_div(
        F.log_softmax(s2, dim=1),
        F.softmax(t2, dim=1),
        reduction="batchmean"
    )

    return (1.0 - alpha) * hard_loss + alpha * (T ** 2) * soft_loss
