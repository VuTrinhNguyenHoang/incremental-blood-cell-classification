from collections.abc import Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import ResNet
from tqdm.auto import tqdm

from incremental_blood_cell.evaluator import evaluate_tasks
from incremental_blood_cell.metrics import (
    average_forgetting,
    final_average_accuracy,
)
from incremental_blood_cell.model import expand_classifier
from incremental_blood_cell.training import train


def run_finetuning(
    model: ResNet,
    class_splits: Sequence[Sequence[int]],
    train_datasets: Sequence[Dataset],
    test_datasets: Sequence[Dataset],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float = 1e-4,
    show_progress: bool = True,
) -> tuple[tuple[float, ...], ...]:
    model.to(device)
    accuracy_matrix = []

    for experience_index, train_dataset in enumerate(train_datasets):
        classes = class_splits[experience_index]

        if experience_index > 0:
            seen_class_count = sum(
                len(class_split) for class_split in class_splits[: experience_index + 1]
            )
            expand_classifier(model, num_classes=seen_class_count)

        if show_progress:
            tqdm.write(
                f"Experience {experience_index + 1}/{len(train_datasets)} "
                f"| classes={tuple(classes)}"
            )

        loader = DataLoader(
            train_dataset,
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
            )

            if experience_index > 0:
                message += f" | forgetting={average_forgetting(accuracy_matrix):.4f}"

            tqdm.write(message)

    return tuple(accuracy_matrix)
