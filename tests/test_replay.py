from collections import Counter
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from incremental_blood_cell.replay import ReplayBuffer, run_random_replay


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 4 * 4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs.flatten(start_dim=1))


def make_dataset(classes: tuple[int, ...]) -> TensorDataset:
    images = []
    labels = []

    for class_id in classes:
        for _ in range(50):
            images.append(torch.full((3, 4, 4), float(class_id)))
            labels.append(class_id)

    return TensorDataset(torch.stack(images), torch.tensor(labels))


def test_replay_buffer_stays_balanced() -> None:
    buffer = ReplayBuffer(capacity=160, seed=0)

    buffer.update(make_dataset((0, 1, 2, 3)), seen_classes=(0, 1, 2, 3))

    assert len(buffer) == 160
    assert buffer.class_counts() == {0: 40, 1: 40, 2: 40, 3: 40}

    buffer.update(make_dataset((4, 5)), seen_classes=(0, 1, 2, 3, 4, 5))

    assert len(buffer) == 160
    assert buffer.class_counts() == {
        0: 27,
        1: 27,
        2: 27,
        3: 27,
        4: 26,
        5: 26,
    }

    buffer.update(make_dataset((6, 7)), seen_classes=tuple(range(8)))

    assert len(buffer) == 160
    assert buffer.class_counts() == {class_id: 20 for class_id in range(8)}


def test_runs_random_replay_with_old_samples() -> None:
    model = TinyClassifier()
    first_task = make_dataset((0, 1))
    second_task = make_dataset((2,))

    training_counts = []

    def record_training_data(**kwargs) -> tuple[float, ...]:
        labels = [int(label) for _, label in kwargs["loader"].dataset]
        training_counts.append(Counter(labels))
        return (0.0,)

    with patch(
        "incremental_blood_cell.replay.train",
        side_effect=record_training_data,
    ):
        accuracy_matrix = run_random_replay(
            model=model,
            class_splits=((0, 1), (2,)),
            train_datasets=(first_task, second_task),
            test_datasets=(first_task, second_task),
            device=torch.device("cpu"),
            epochs=1,
            batch_size=16,
            learning_rate=0.1,
            memory_size=3,
            show_progress=False,
        )

    assert training_counts == [
        Counter({0: 50, 1: 50}),
        Counter({2: 50, 0: 2, 1: 1}),
    ]

    assert len(accuracy_matrix) == 2
    assert len(accuracy_matrix[0]) == 1
    assert len(accuracy_matrix[1]) == 2
    assert model.fc.out_features == 3
