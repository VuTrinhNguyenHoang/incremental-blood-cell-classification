from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def evaluate_tasks(
    model: nn.Module, datasets: Sequence[Dataset], batch_size: int = 128
) -> tuple[float, ...]:
    device = next(model.parameters()).device
    was_training = model.training

    model.eval()
    accuracies = []

    with torch.inference_mode():
        for dataset in datasets:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            correct = 0
            total = 0

            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                predictions = model(images).argmax(dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.numel()

            accuracies.append(correct / total)

    model.train(was_training)

    return tuple(accuracies)
