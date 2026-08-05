from pathlib import Path

from medmnist import BloodMNIST
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
