import torch
from torch import nn
from torch.utils.data import TensorDataset

from incremental_blood_cell.methods.selection import (
    collect_features_and_logits,
    select_exemplars,
)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


def test_selects_sample_closest_to_prototype() -> None:
    features = torch.tensor(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )
    logits = torch.zeros(3, 2)
    labels = torch.tensor([0, 0, 0])

    selected = select_exemplars(
        features=features,
        logits=logits,
        labels=labels,
        quotas={0: 1},
        strategy="prototype",
    )

    assert selected == (1,)


def test_selects_sample_with_smallest_margin() -> None:
    features = torch.zeros(3, 1)
    logits = torch.tensor(
        [
            [5.0, 0.0],
            [1.1, 1.0],
            [3.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 0, 0])

    selected = select_exemplars(
        features=features,
        logits=logits,
        labels=labels,
        quotas={0: 1},
        strategy="boundary",
    )

    assert selected == (1,)


def test_hybrid_selection_avoids_duplicates() -> None:
    features = torch.tensor(
        [
            [0.0],
            [1.0],
            [2.0],
            [10.0],
        ]
    )
    logits = torch.tensor(
        [
            [4.0, 0.0],
            [1.2, 1.0],
            [1.01, 1.0],
            [5.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 0, 0, 0])

    selected = select_exemplars(
        features=features,
        logits=logits,
        labels=labels,
        quotas={0: 2},
        strategy="hybrid",
    )

    assert selected == (2, 1)
    assert len(set(selected)) == 2


def test_collects_features_and_logits_in_dataset_order() -> None:
    model = TinyClassifier()

    with torch.no_grad():
        model.fc.weight.copy_(torch.eye(2))

    images = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    labels = torch.tensor([0, 1, 0])
    dataset = TensorDataset(images, labels)

    model.train()

    features, logits, collected_labels = collect_features_and_logits(
        model=model,
        dataset=dataset,
        batch_size=2,
    )

    assert torch.equal(features, images)
    assert torch.equal(logits, images)
    assert torch.equal(collected_labels, labels)
    assert model.training
