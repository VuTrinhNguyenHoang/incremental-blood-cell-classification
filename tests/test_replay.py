import torch
from torch.utils.data import TensorDataset

from incremental_blood_cell.replay import ReplayBuffer


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
