from collections.abc import Sequence
from copy import deepcopy

import torch
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.models.resnet import ResNet
from tqdm.auto import tqdm

from incremental_blood_cell.evaluator import evaluate_tasks
from incremental_blood_cell.lwf import train_lwf
from incremental_blood_cell.metrics import (
    average_forgetting,
    final_average_accuracy,
)
from incremental_blood_cell.model import expand_classifier
from incremental_blood_cell.replay import ReplayBuffer
from incremental_blood_cell.training import train


def run_replay_kd(
    model: ResNet,
    class_splits: Sequence[Sequence[int]],
    train_datasets: Sequence[Dataset],
    test_datasets: Sequence[Dataset],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    memory_size: int,
    distillation_weight: float,
    temperature: float,
    seed: int = 0,
    weight_decay: float = 1e-4,
    show_progress: bool = True,
) -> tuple[tuple[float, ...], ...]:
    model.to(device)

    buffer = ReplayBuffer(capacity=memory_size, seed=seed)
    accuracy_matrix = []

    for experience_index, train_dataset in enumerate(train_datasets):
        classes = class_splits[experience_index]
        seen_classes = tuple(
            class_id
            for class_split in class_splits[: experience_index + 1]
            for class_id in class_split
        )
        teacher = None

        if experience_index > 0:
            teacher = deepcopy(model)
            expand_classifier(model, num_classes=len(seen_classes))

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

        if teacher is None:
            losses = train(
                model=model,
                loader=loader,
                optimizer=optimizer,
                epochs=epochs,
                show_progress=show_progress,
            )
        else:
            losses = train_lwf(
                student=model,
                teacher=teacher,
                loader=loader,
                optimizer=optimizer,
                epochs=epochs,
                distillation_weight=distillation_weight,
                temperature=temperature,
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
