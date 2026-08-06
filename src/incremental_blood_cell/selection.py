from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

SelectionStrategy = Literal["prototype", "boundary", "hybrid"]


def _collate_samples(
    samples: list[tuple[torch.Tensor, int | torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = zip(*samples)

    return (
        torch.stack(images),
        torch.tensor([int(label) for label in labels]),
    )


def collect_features_and_logits(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_samples,
    )

    feature_batches = []
    logit_batches = []
    label_batches = []

    def capture_features(
        _module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        feature_batches.append(inputs[0].detach().cpu())

    handle = model.fc.register_forward_pre_hook(capture_features)
    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            for images, labels in loader:
                logits = model(images.to(device))

                logit_batches.append(logits.cpu())
                label_batches.append(labels.cpu())
    finally:
        handle.remove()
        model.train(was_training)

    return (
        torch.cat(feature_batches),
        torch.cat(logit_batches),
        torch.cat(label_batches),
    )


def select_exemplars(
    features: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    quotas: dict[int, int],
    strategy: SelectionStrategy,
) -> tuple[int, ...]:
    selected_indices = []

    for class_id, quota in quotas.items():
        class_indices = torch.where(labels == class_id)[0]

        if quota <= 0 or len(class_indices) == 0:
            continue

        sample_count = min(quota, len(class_indices))
        class_features = features[class_indices]
        class_logits = logits[class_indices]

        prototype_order = _prototype_order(class_features)
        boundary_order = _boundary_order(class_logits)

        if strategy == "prototype":
            local_indices = prototype_order[:sample_count]
        elif strategy == "boundary":
            local_indices = boundary_order[:sample_count]
        elif strategy == "hybrid":
            prototype_count = (sample_count + 1) // 2
            local_indices = prototype_order[:prototype_count]

            boundary_indices = [
                index for index in boundary_order if index not in local_indices
            ]
            local_indices.extend(boundary_indices[: sample_count - prototype_count])
        else:
            raise ValueError(f"unknown selection strategy: {strategy}")

        selected_indices.extend(class_indices[local_indices].tolist())

    return tuple(selected_indices)


def _prototype_order(features: torch.Tensor) -> list[int]:
    centroid = features.mean(dim=0, keepdim=True)
    distances = torch.linalg.vector_norm(
        features - centroid,
        dim=1,
    )
    return torch.argsort(distances).tolist()


def _boundary_order(logits: torch.Tensor) -> list[int]:
    probabilities = F.softmax(logits, dim=1)
    top_probabilities = probabilities.topk(k=2, dim=1).values
    margins = top_probabilities[:, 0] - top_probabilities[:, 1]

    return torch.argsort(margins).tolist()
