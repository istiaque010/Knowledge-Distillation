"""Phase 7: the knowledge-distillation loss.

    L = alpha * L_CE  +  (1 - alpha) * T^2 * L_KD

L_CE  hard-label loss: normal next-token cross-entropy against the real TinyStories token
L_KD  soft-target loss: KL(p_teacher || p_student), both softened with temperature T:
          p_teacher = softmax(z_teacher / T),   p_student = softmax(z_student / T)
alpha balances "learn from the real data" (alpha = 1: normal training, no teacher) against "imitate the teacher".
T^2   softening by T shrinks the KD gradients by about 1/T^2; multiplying by T^2 keeps them at the same scale, so
      changing T changes WHAT is learned (how much the second-best guesses matter), not HOW MUCH.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """KL(p_teacher || p_student) at temperature T, averaged over token positions."""
    V = student_logits.size(-1)
    log_ps = F.log_softmax(student_logits.reshape(-1, V) / temperature, dim=-1)
    log_pt = F.log_softmax(teacher_logits.reshape(-1, V) / temperature, dim=-1)
    # sum over the vocabulary of p_t * (log p_t - log p_s), then the mean over positions ("batchmean" on 2-D input)
    return F.kl_div(log_ps, log_pt, log_target=True, reduction="batchmean")


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor | None, targets: torch.Tensor,
                      temperature: float = 2.0, alpha: float = 0.5):
    """Returns (total, ce, kd). ce and kd are detached copies, for logging.

    student_logits, teacher_logits: (batch, length, vocab); targets: (batch, length) real next tokens.
    teacher_logits=None or alpha=1 gives plain cross-entropy (the student-from-scratch baseline).
    """
    V = student_logits.size(-1)
    ce = F.cross_entropy(student_logits.reshape(-1, V), targets.reshape(-1))
    if teacher_logits is None or alpha >= 1.0:
        return ce, ce.detach(), torch.zeros((), device=ce.device)
    kd = kd_kl(student_logits, teacher_logits, temperature)
    total = alpha * ce + (1.0 - alpha) * temperature**2 * kd
    return total, ce.detach(), kd.detach()
