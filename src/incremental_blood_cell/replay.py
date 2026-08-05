import random
from collections import Counter
from collections.abc import Sequence

import torch
from torch.utils.data import Dataset


class ReplayBuffer(Dataset):
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = capacity
        self._random = random.Random(seed)
        self._samples: list[tuple[torch.Tensor, int]] = []

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self._samples[index]

    def class_counts(self) -> dict[int, int]:
        return dict(Counter(label for _, label in self._samples))

    def update(self, dataset: Dataset, seen_classes: Sequence[int]) -> None:
        base, remainder = divmod(self.capacity, len(seen_classes))
        quotas = {
            class_id: base + (index < remainder)
            for index, class_id in enumerate(seen_classes)
        }

        old_samples: dict[int, list[tuple[torch.Tensor, int]]] = {}
        for sample in self._samples:
            old_samples.setdefault(sample[1], []).append(sample)

        retained = []
        for class_id, samples in old_samples.items():
            retained.extend(
                self._random.sample(samples, min(quotas[class_id], len(samples)))
            )

        new_classes = {
            class_id for class_id in seen_classes if class_id not in old_samples
        }
        selected = self._select(dataset, {c: quotas[c] for c in new_classes})

        self._samples = retained
        for class_id in seen_classes:
            self._samples.extend(selected.get(class_id, []))

    def _select(
        self, dataset: Dataset, quotas: dict[int, int]
    ) -> dict[int, list[tuple[torch.Tensor, int]]]:
        selected = {class_id: [] for class_id in quotas}
        seen_counts = Counter()

        for image, label in dataset:
            label = int(label)

            if label not in quotas:
                continue

            seen_counts[label] += 1
            quota = quotas[label]
            samples = selected[label]

            if len(samples) < quota:
                samples.append((image.detach().cpu().clone(), label))
                continue

            position = self._random.randrange(seen_counts[label])
            if position < quota:
                samples[position] = (image.detach().cpu().clone(), label)

        return selected
