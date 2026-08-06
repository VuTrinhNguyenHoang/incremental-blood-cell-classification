from collections.abc import Sequence
from pathlib import Path

from medmnist import BloodMNIST
from torch.utils.data import Dataset, Subset
from torchvision.transforms import ToTensor


def _class_index(target) -> int:
    return int(target[0])


class RemappedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        label_map: dict[int, int],
    ) -> None:
        self.dataset = dataset
        self.label_map = dict(label_map)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, self.label_map[int(label)]


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


def build_experience_datasets(
    dataset: BloodMNIST,
    class_splits: Sequence[Sequence[int]],
) -> tuple[RemappedDataset, ...]:
    class_order = [class_id for class_split in class_splits for class_id in class_split]
    label_map = {class_id: new_label for new_label, class_id in enumerate(class_order)}

    return tuple(
        RemappedDataset(
            dataset=subset_by_classes(dataset, classes),
            label_map=label_map,
        )
        for classes in class_splits
    )
