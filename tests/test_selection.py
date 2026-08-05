import torch

from incremental_blood_cell.selection import select_exemplars


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
