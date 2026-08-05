import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange


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
