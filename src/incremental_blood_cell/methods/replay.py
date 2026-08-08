import random
from collections import Counter
from collections.abc import Sequence

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.models.resnet import ResNet
from tqdm.auto import tqdm

from incremental_blood_cell.evaluator import evaluate_tasks
from incremental_blood_cell.methods.selection import (
    SelectionStrategy,
    collect_features_and_logits,
    select_exemplars,
)
from incremental_blood_cell.metrics import (
    average_forgetting,
    final_average_accuracy,
)
from incremental_blood_cell.model import expand_classifier
from incremental_blood_cell.training import train


def _balanced_quotas(
    capacity: int,
    classes: Sequence[int],
) -> dict[int, int]:
    base, remainder = divmod(capacity, len(classes))

    return {
        class_id: base + (index < remainder) for index, class_id in enumerate(classes)
    }


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
        quotas = _balanced_quotas(self.capacity, seen_classes)

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


class SelectionReplayBuffer(Dataset):
    def __init__(
        self,
        capacity: int,
        strategy: SelectionStrategy = "hybrid",
    ) -> None:
        self.capacity = capacity
        self.strategy = strategy
        self._samples: list[tuple[torch.Tensor, int]] = []

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self._samples[index]

    def class_counts(self) -> dict[int, int]:
        return dict(Counter(label for _, label in self._samples))

    def update(
        self,
        model: nn.Module,
        dataset: Dataset,
        seen_classes: Sequence[int],
        batch_size: int,
    ) -> None:
        if self.capacity == 0:
            self._samples = []
            return

        candidates = dataset
        if len(self) > 0:
            candidates = ConcatDataset((dataset, self))

        quotas = _balanced_quotas(
            capacity=self.capacity,
            classes=seen_classes,
        )

        features, logits, labels = collect_features_and_logits(
            model=model,
            dataset=candidates,
            batch_size=batch_size,
        )

        selected_indices = select_exemplars(
            features=features,
            logits=logits,
            labels=labels,
            quotas=quotas,
            strategy=self.strategy,
        )

        selected_samples = []

        for index in selected_indices:
            image, label = candidates[index]
            selected_samples.append(
                (
                    image.detach().cpu().clone(),
                    int(label),
                )
            )

        self._samples = selected_samples


def run_random_replay(
    model: ResNet,
    class_splits: Sequence[Sequence[int]],
    train_datasets: Sequence[Dataset],
    test_datasets: Sequence[Dataset],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    memory_size: int,
    seed: int = 0,
    weight_decay: float = 1e-4,
    show_progress: bool = True,
) -> tuple[tuple[float, ...], ...]:
    model.to(device)

    buffer = ReplayBuffer(capacity=memory_size, seed=seed)
    accuracy_matrix = []

    for experience_index, train_dataset in enumerate(train_datasets):
        classes = class_splits[experience_index]
        seen_class_count = sum(
            len(class_split) for class_split in class_splits[: experience_index + 1]
        )
        seen_classes = tuple(range(seen_class_count))

        if experience_index > 0:
            expand_classifier(
                model,
                num_classes=len(seen_classes),
            )

        if show_progress:
            tqdm.write(
                f"Experience {experience_index + 1}/{len(train_datasets)} "
                f"| classes={tuple(classes)}"
            )

        training_dataset = train_dataset
        if len(buffer) > 0:
            training_dataset = ConcatDataset((train_dataset, buffer))

        loader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        losses = train(
            model=model,
            loader=loader,
            optimizer=optimizer,
            epochs=epochs,
            show_progress=show_progress,
        )

        buffer.update(
            dataset=train_dataset,
            seen_classes=seen_classes,
        )

        row = evaluate_tasks(
            model=model,
            datasets=test_datasets[: experience_index + 1],
            batch_size=batch_size,
        )
        accuracy_matrix.append(row)

        if show_progress:
            message = (
                f"Experience {experience_index + 1}/{len(train_datasets)}"
                f" | loss={losses[-1]:.4f}"
                f" | avg_acc={final_average_accuracy(accuracy_matrix):.4f}"
                f" | memory={len(buffer)}"
            )

            if experience_index > 0:
                message += f" | forgetting={average_forgetting(accuracy_matrix):.4f}"

            tqdm.write(message)

    return tuple(accuracy_matrix)
