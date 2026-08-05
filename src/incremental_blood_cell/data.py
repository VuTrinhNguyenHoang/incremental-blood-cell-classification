from collections.abc import Sequence
from pathlib import Path

from medmnist import BloodMNIST
from torch.utils.data import Subset
from torchvision.transforms import ToTensor


def _class_index(target) -> int:
    return int(target[0])


def load_bloodmnist(
    split: str, root: str | Path = "data", download: bool = False
) -> BloodMNIST:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    return BloodMNIST(
        split=split,
        root=str(root),
        size=64,
        download=download,
        transform=ToTensor(),
        target_transform=_class_index,
    )


def subset_by_classes(dataset: BloodMNIST, classes: Sequence[int]) -> Subset:
    class_set = set(classes)
    indices = [
        i for i, target in enumerate(dataset.labels) if int(target[0]) in class_set
    ]
    return Subset(dataset, indices)
