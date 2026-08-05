from pathlib import Path
from unittest.mock import patch

from incremental_blood_cell.data import load_bloodmnist


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
