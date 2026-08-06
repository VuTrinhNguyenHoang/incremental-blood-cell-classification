from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from incremental_blood_cell.methods.lwf import distillation_loss, run_lwf, train_lwf


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


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


def test_runs_sequential_lwf_with_old_teacher() -> None:
    model = TinyClassifier()

    first_task = TensorDataset(
        torch.tensor(
            [
                [-2.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ]
        ),
        torch.tensor([0, 0, 1, 1]),
    )

    second_task = TensorDataset(
        torch.tensor(
            [
                [0.0, 1.0],
                [0.0, 2.0],
            ]
        ),
        torch.tensor([2, 2]),
    )

    with patch(
        "incremental_blood_cell.methods.lwf.train_lwf",
        wraps=train_lwf,
    ) as mocked_train_lwf:
        accuracy_matrix = run_lwf(
            model=model,
            class_splits=((0, 1), (2,)),
            train_datasets=(first_task, second_task),
            test_datasets=(first_task, second_task),
            device=torch.device("cpu"),
            epochs=1,
            batch_size=2,
            learning_rate=0.1,
            distillation_weight=1.0,
            temperature=2.0,
            show_progress=False,
        )

    teacher = mocked_train_lwf.call_args.kwargs["teacher"]

    assert mocked_train_lwf.call_count == 1
    assert teacher is not model
    assert teacher.fc.out_features == 2
    assert model.fc.out_features == 3

    assert len(accuracy_matrix) == 2
    assert len(accuracy_matrix[0]) == 1
    assert len(accuracy_matrix[1]) == 2

    assert all(0.0 <= accuracy <= 1.0 for row in accuracy_matrix for accuracy in row)
