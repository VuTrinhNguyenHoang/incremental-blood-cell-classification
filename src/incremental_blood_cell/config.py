from dataclasses import dataclass
from typing import Literal

from incremental_blood_cell.scenario import build_class_splits

Method = Literal[
    "joint",
    "finetuning",
    "lwf",
    "random_replay",
    "replay_kd",
    "prototype",
    "boundary",
    "hybrid",
]


@dataclass(frozen=True)
class ExperimentConfig:
    method: Method
    class_order: tuple[int, ...] = tuple(range(8))
    increments: tuple[int, ...] = (4, 2, 2)
    seed: int = 0
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    memory_size: int = 160
    distillation_weight: float = 1.0
    temperature: float = 2.0

    @property
    def class_splits(self) -> tuple[tuple[int, ...], ...]:
        return build_class_splits(
            class_order=self.class_order,
            increments=self.increments,
        )
