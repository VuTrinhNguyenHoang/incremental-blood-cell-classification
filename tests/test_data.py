from pathlib import Path
from unittest.mock import patch

from incremental_blood_cell.data import (
    build_experience_datasets,
    load_bloodmnist,
    subset_by_classes,
)
from incremental_blood_cell.scenario import build_class_splits


class DummyDataset:
    def __init__(self) -> None:
        self.labels = [[class_id] for class_id in range(8) for _ in range(2)]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[int, int]:
        return index, int(self.labels[index][0])


def test_loads_bloodmnist(tmp_path: Path) -> None:
    root = tmp_path / "data"

    with patch("incremental_blood_cell.data.BloodMNIST") as dataset_class:
        load_bloodmnist(split="train", root=root, download=False)

    arguments = dataset_class.call_args.kwargs

    assert root.exists()
    assert arguments["split"] == "train"
    assert arguments["root"] == str(root)
    assert arguments["size"] == 64
    assert arguments["download"] is False
    assert callable(arguments["transform"])
    assert arguments["target_transform"]([3]) == 3


def test_experience_has_no_future_classes() -> None:
    dataset = DummyDataset()
    class_splits = build_class_splits(class_order=range(8), increments=(4, 2, 2))

    experience = subset_by_classes(dataset=dataset, classes=class_splits[0])

    labels = {experience[i][1] for i in range(len(experience))}

    assert len(experience) == 8
    assert labels == {0, 1, 2, 3}


def test_remaps_labels_for_arbitrary_class_order() -> None:
    dataset = DummyDataset()
    class_order = (4, 6, 1, 7, 0, 3, 2, 5)
    class_splits = build_class_splits(
        class_order=class_order,
        increments=(4, 2, 2),
    )

    experiences = build_experience_datasets(
        dataset=dataset,
        class_splits=class_splits,
    )

    observed_mapping = {}

    for experience in experiences:
        for index in range(len(experience)):
            image_index, remapped_label = experience[index]
            original_label = int(dataset.labels[image_index][0])
            observed_mapping[original_label] = remapped_label

    assert observed_mapping == {
        4: 0,
        6: 1,
        1: 2,
        7: 3,
        0: 4,
        3: 5,
        2: 6,
        5: 7,
    }

    assert [
        {experience[i][1] for i in range(len(experience))} for experience in experiences
    ] == [
        {0, 1, 2, 3},
        {4, 5},
        {6, 7},
    ]
