import torch
from torch.utils.data import Dataset

from incremental_blood_cell.config import ExperimentConfig
from incremental_blood_cell.data import build_experience_datasets
from incremental_blood_cell.finetuning import run_finetuning
from incremental_blood_cell.joint import run_joint_training
from incremental_blood_cell.lwf import run_lwf
from incremental_blood_cell.model import build_resnet18
from incremental_blood_cell.replay import run_random_replay
from incremental_blood_cell.replay_kd import run_replay_kd
from incremental_blood_cell.selection_replay import run_selection_replay_kd
from incremental_blood_cell.training import set_seed


def run_experiment(
    config: ExperimentConfig,
    train_dataset: Dataset,
    test_dataset: Dataset,
    device: torch.device,
    show_progress: bool = True,
) -> tuple[tuple[float, ...], ...]:
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
        return run_joint_training(**common_arguments)

    if config.method == "finetuning":
        return run_finetuning(**common_arguments)

    if config.method == "lwf":
        return run_lwf(
            **common_arguments,
            distillation_weight=config.distillation_weight,
            temperature=config.temperature,
        )

    if config.method == "random_replay":
        return run_random_replay(
            **common_arguments,
            memory_size=config.memory_size,
            seed=config.seed,
        )

    if config.method == "replay_kd":
        return run_replay_kd(
            **common_arguments,
            memory_size=config.memory_size,
            distillation_weight=config.distillation_weight,
            temperature=config.temperature,
            seed=config.seed,
        )

    if config.method in {"prototype", "boundary", "hybrid"}:
        return run_selection_replay_kd(
            **common_arguments,
            memory_size=config.memory_size,
            selection_strategy=config.method,
            distillation_weight=config.distillation_weight,
            temperature=config.temperature,
        )

    raise ValueError(f"unknown method: {config.method}")
