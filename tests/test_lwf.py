import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from incremental_blood_cell.lwf import distillation_loss, train_lwf


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


def test_lwf_updates_student_but_not_teacher() -> None:
    teacher = nn.Linear(
        in_features=2,
        out_features=2,
    )
    student = nn.Linear(
        in_features=2,
        out_features=3,
    )

    dataset = TensorDataset(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        ),
        torch.tensor([2, 2, 2, 2]),
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    optimizer = SGD(
        student.parameters(),
        lr=0.1,
    )

    old_teacher_parameters = [
        parameter.detach().clone() for parameter in teacher.parameters()
    ]
    old_student_parameters = [
        parameter.detach().clone() for parameter in student.parameters()
    ]

    teacher.train()

    losses = train_lwf(
        student=student,
        teacher=teacher,
        loader=loader,
        optimizer=optimizer,
        epochs=2,
        distillation_weight=1.0,
        temperature=2.0,
        show_progress=False,
    )

    teacher_unchanged = all(
        torch.equal(old, new)
        for old, new in zip(
            old_teacher_parameters,
            teacher.parameters(),
            strict=True,
        )
    )
    student_changed = any(
        not torch.equal(old, new)
        for old, new in zip(
            old_student_parameters,
            student.parameters(),
            strict=True,
        )
    )

    assert len(losses) == 2
    assert all(loss > 0.0 for loss in losses)
    assert teacher_unchanged
    assert student_changed
    assert not teacher.training
    assert all(parameter.grad is None for parameter in teacher.parameters())
