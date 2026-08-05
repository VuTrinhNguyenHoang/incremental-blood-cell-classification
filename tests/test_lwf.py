import pytest
import torch

from incremental_blood_cell.lwf import distillation_loss


def test_distillation_uses_only_old_class_logits() -> None:
    teacher_logits = torch.tensor(
        [
            [2.0, 1.0],
            [1.0, 3.0],
        ],
        requires_grad=True,
    )

    student_logits = torch.tensor(
        [
            [2.0, 1.0, 10.0],
            [1.0, 3.0, -10.0],
        ],
        requires_grad=True,
    )

    loss = distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        temperature=2.0,
    )

    loss.backward()

    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert student_logits.grad is not None
    assert teacher_logits.grad is None


def test_distillation_requires_positive_temperature() -> None:
    logits = torch.tensor([[1.0, 2.0]])

    with pytest.raises(ValueError, match="positive"):
        distillation_loss(
            student_logits=logits,
            teacher_logits=logits,
            temperature=0.0,
        )
