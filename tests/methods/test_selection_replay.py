from collections import Counter
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from incremental_blood_cell.methods.selection_replay import run_selection_replay_kd


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


def test_runs_selection_replay_with_memory_and_teacher() -> None:
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

    observed = {}

    def inspect_training(**kwargs) -> tuple[float, ...]:
        labels = [int(label) for _, label in kwargs["loader"].dataset]

        observed["counts"] = Counter(labels)
        observed["teacher_classes"] = kwargs["teacher"].fc.out_features
        observed["student_classes"] = kwargs["student"].fc.out_features

        return (0.0,)

    with patch(
        "incremental_blood_cell.methods.selection_replay.train_lwf",
        side_effect=inspect_training,
    ) as mocked_train_lwf:
        accuracy_matrix = run_selection_replay_kd(
            model=model,
            class_splits=((0, 1), (2,)),
            train_datasets=(first_task, second_task),
            test_datasets=(first_task, second_task),
            device=torch.device("cpu"),
            epochs=1,
            batch_size=2,
            learning_rate=0.1,
            memory_size=3,
            selection_strategy="hybrid",
            distillation_weight=1.0,
            temperature=2.0,
            show_progress=False,
        )

    assert mocked_train_lwf.call_count == 1
    assert observed["counts"] == Counter({0: 2, 1: 1, 2: 2})
    assert observed["teacher_classes"] == 2
    assert observed["student_classes"] == 3

    assert len(accuracy_matrix) == 2
    assert len(accuracy_matrix[0]) == 1
    assert len(accuracy_matrix[1]) == 2
    assert model.fc.out_features == 3
