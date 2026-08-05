import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    show_progress: bool = True,
) -> float:
    device = next(model.parameters()).device

    model.train()

    total_loss = 0.0
    total_samples = 0

    batches = tqdm(loader, desc="Batches", leave=False, disable=not show_progress)

    for images, labels in batches:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        running_loss = total_loss / total_samples
        batches.set_postfix(loss=f"{running_loss:.4f}")

    return total_loss / total_samples


def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    epochs: int,
    show_progress: bool = True,
) -> tuple[float, ...]:
    losses = []

    epoch_progress = trange(epochs, desc="Training", disable=not show_progress)

    for _ in epoch_progress:
        loss = train_one_epoch(model, loader, optimizer, show_progress=show_progress)

        losses.append(loss)
        epoch_progress.set_postfix(loss=f"{loss:.4f}")

    return tuple(losses)
