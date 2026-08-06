from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from torch.utils.data import Dataset

from incremental_blood_cell.config import ExperimentConfig
from incremental_blood_cell.data import build_experience_datasets
from incremental_blood_cell.finetuning import run_finetuning
from incremental_blood_cell.joint import run_joint_training
from incremental_blood_cell.lwf import run_lwf
from incremental_blood_cell.metrics import (
    average_forgetting,
    backward_transfer,
    final_average_accuracy,
)
from incremental_blood_cell.model import build_resnet18
from incremental_blood_cell.replay import run_random_replay
from incremental_blood_cell.replay_kd import run_replay_kd
from incremental_blood_cell.selection_replay import run_selection_replay_kd
from incremental_blood_cell.training import set_seed


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    accuracy_matrix: tuple[tuple[float, ...], ...]
    model: nn.Module = field(repr=False)

    @property
    def final_average_accuracy(self) -> float:
        return final_average_accuracy(self.accuracy_matrix)

    @property
    def average_forgetting(self) -> float:
        return average_forgetting(self.accuracy_matrix)

    @property
    def backward_transfer(self) -> float:
        return backward_transfer(self.accuracy_matrix)

    def to_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "accuracy_matrix": [list(row) for row in self.accuracy_matrix],
            "final_average_accuracy": self.final_average_accuracy,
            "average_forgetting": self.average_forgetting,
            "backward_transfer": self.backward_transfer,
        }


def run_experiment(
    config: ExperimentConfig,
    train_dataset: Dataset,
    test_dataset: Dataset,
    device: torch.device,
    show_progress: bool = True,
) -> ExperimentResult:
    set_seed(config.seed)

    class_splits = config.class_splits
    train_datasets = build_experience_datasets(
        dataset=train_dataset,
        class_splits=class_splits,
    )
    test_datasets = build_experience_datasets(
        dataset=test_dataset,
        class_splits=class_splits,
    )

    model = build_resnet18(
        num_classes=len(class_splits[0]),
    )

    common_arguments = {
        "model": model,
        "class_splits": class_splits,
        "train_datasets": train_datasets,
        "test_datasets": test_datasets,
        "device": device,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "show_progress": show_progress,
    }

    if config.method == "joint":
        accuracy_matrix = run_joint_training(**common_arguments)

    elif config.method == "finetuning":
        accuracy_matrix = run_finetuning(**common_arguments)

    elif config.method == "lwf":
        accuracy_matrix = run_lwf(
            **common_arguments,
            distillation_weight=config.distillation_weight,
            temperature=config.temperature,
        )

    elif config.method == "random_replay":
        accuracy_matrix = run_random_replay(
            **common_arguments,
            memory_size=config.memory_size,
            seed=config.seed,
        )

    elif config.method == "replay_kd":
        accuracy_matrix = run_replay_kd(
            **common_arguments,
            memory_size=config.memory_size,
            distillation_weight=config.distillation_weight,
            temperature=config.temperature,
            seed=config.seed,
        )

    elif config.method in {"prototype", "boundary", "hybrid"}:
        accuracy_matrix = run_selection_replay_kd(
            **common_arguments,
            memory_size=config.memory_size,
            selection_strategy=config.method,
            distillation_weight=config.distillation_weight,
            temperature=config.temperature,
        )

    else:
        raise ValueError(f"unknown method: {config.method}")

    return ExperimentResult(
        config=config,
        accuracy_matrix=accuracy_matrix,
        model=model,
    )
