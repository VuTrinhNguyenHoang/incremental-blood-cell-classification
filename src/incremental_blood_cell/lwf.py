from collections.abc import Sequence
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import ResNet
from tqdm.auto import tqdm, trange

from incremental_blood_cell.evaluator import evaluate_tasks
from incremental_blood_cell.metrics import (
    average_forgetting,
    final_average_accuracy,
)
from incremental_blood_cell.model import expand_classifier
from incremental_blood_cell.training import train


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    old_class_count = teacher_logits.size(1)

    student_log_probabilities = F.log_softmax(
        student_logits[:, :old_class_count] / temperature,
        dim=1,
    )

    teacher_probabilities = F.softmax(
        teacher_logits.detach() / temperature,
        dim=1,
    )

    return (
        F.kl_div(
            student_log_probabilities,
            teacher_probabilities,
            reduction="batchmean",
        )
        * temperature**2
    )


def train_lwf_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    distillation_weight: float,
    temperature: float,
    show_progress: bool = True,
) -> float:
    device = next(student.parameters()).device

    teacher.to(device)
    teacher.eval()
    teacher.requires_grad_(False)
    teacher.zero_grad(set_to_none=True)

    student.train()

    total_loss = 0.0
    total_classification_loss = 0.0
    total_distillation_loss = 0.0
    total_samples = 0

    batches = tqdm(
        loader,
        desc="Batches",
        leave=False,
        disable=not show_progress,
    )

    for images, labels in batches:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.inference_mode():
            teacher_logits = teacher(images)

        student_logits = student(images)

        classification_loss = F.cross_entropy(
            student_logits,
            labels,
        )
        knowledge_distillation_loss = distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            temperature=temperature,
        )

        loss = classification_loss + distillation_weight * knowledge_distillation_loss

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_classification_loss += classification_loss.item() * batch_size
        total_distillation_loss += knowledge_distillation_loss.item() * batch_size
        total_samples += batch_size

        batches.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            ce=f"{total_classification_loss / total_samples:.4f}",
            kd=f"{total_distillation_loss / total_samples:.4f}",
        )

    return total_loss / total_samples


def train_lwf(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    epochs: int,
    distillation_weight: float,
    temperature: float,
    show_progress: bool = True,
) -> tuple[float, ...]:
    losses = []

    epoch_progress = trange(
        epochs,
        desc="Training",
        disable=not show_progress,
    )

    for _ in epoch_progress:
        loss = train_lwf_one_epoch(
            student=student,
            teacher=teacher,
            loader=loader,
            optimizer=optimizer,
            distillation_weight=distillation_weight,
            temperature=temperature,
            show_progress=show_progress,
        )

        losses.append(loss)
        epoch_progress.set_postfix(loss=f"{loss:.4f}")

    return tuple(losses)


def run_lwf(
    model: ResNet,
    class_splits: Sequence[Sequence[int]],
    train_datasets: Sequence[Dataset],
    test_datasets: Sequence[Dataset],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    distillation_weight: float,
    temperature: float,
    weight_decay: float = 1e-4,
    show_progress: bool = True,
) -> tuple[tuple[float, ...], ...]:
    model.to(device)
    accuracy_matrix = []

    for experience_index, train_dataset in enumerate(train_datasets):
        classes = class_splits[experience_index]
        teacher = None

        if experience_index > 0:
            teacher = deepcopy(model)

            seen_class_count = sum(
                len(class_split) for class_split in class_splits[: experience_index + 1]
            )
            expand_classifier(
                model,
                num_classes=seen_class_count,
            )

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
