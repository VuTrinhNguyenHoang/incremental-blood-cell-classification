import torch
from torch.nn import functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    old_class_count = teacher_logits.size(1)

    student_log_probabilities = F.log_softmax(
        student_logits[:, :old_class_count] / temperature,
        dim=1,
    )

    teacher_probabilities = F.softmax(
        teacher_logits.detach() / temperature,
        dim=1,
    )

    return (
        F.kl_div(
            student_log_probabilities,
            teacher_probabilities,
            reduction="batchmean",
        )
        * temperature**2
    )
