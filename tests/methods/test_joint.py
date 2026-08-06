from collections import Counter
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from incremental_blood_cell.methods.joint import run_joint_training


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


def test_joint_training_uses_all_seen_data() -> None:
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

    training_counts = []

    def inspect_training(**kwargs) -> tuple[float, ...]:
        labels = [int(label) for _, label in kwargs["loader"].dataset]
        training_counts.append(Counter(labels))
        return (0.0,)

    with patch(
        "incremental_blood_cell.methods.joint.train",
        side_effect=inspect_training,
    ):
        accuracy_matrix = run_joint_training(
            model=model,
            class_splits=((0, 1), (2,)),
            train_datasets=(first_task, second_task),
            test_datasets=(first_task, second_task),
            device=torch.device("cpu"),
            epochs=1,
            batch_size=2,
            learning_rate=0.1,
            show_progress=False,
        )

    assert training_counts == [
        Counter({0: 2, 1: 2}),
        Counter({0: 2, 1: 2, 2: 2}),
    ]

    assert len(accuracy_matrix) == 2
    assert len(accuracy_matrix[0]) == 1
    assert len(accuracy_matrix[1]) == 2
    assert model.fc.out_features == 3
