from unittest.mock import patch

import pytest
import torch

from incremental_blood_cell.config import ExperimentConfig
from incremental_blood_cell.experiment import run_experiment


class DummyDataset:
    def __init__(self) -> None:
        self.labels = [[class_id] for class_id in range(8)]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[int, int]:
        return index, int(self.labels[index][0])


@pytest.mark.parametrize(
    ("method", "runner_name"),
    [
        ("joint", "run_joint_training"),
        ("finetuning", "run_finetuning"),
        ("lwf", "run_lwf"),
        ("random_replay", "run_random_replay"),
        ("replay_kd", "run_replay_kd"),
        ("prototype", "run_selection_replay_kd"),
        ("boundary", "run_selection_replay_kd"),
        ("hybrid", "run_selection_replay_kd"),
    ],
)
def test_dispatches_configured_method(
    method: str,
    runner_name: str,
) -> None:
    dataset = DummyDataset()
    model = object()
    expected_result = ((0.5,),)

    config = ExperimentConfig(
        method=method,
        class_order=(4, 6, 1, 7, 0, 3, 2, 5),
        epochs=1,
    )

    with (
        patch(
            "incremental_blood_cell.experiment.build_resnet18",
            return_value=model,
        ) as mocked_build_model,
        patch(
            f"incremental_blood_cell.experiment.{runner_name}",
            return_value=expected_result,
        ) as mocked_runner,
    ):
        result = run_experiment(
            config=config,
            train_dataset=dataset,
            test_dataset=dataset,
            device=torch.device("cpu"),
            show_progress=False,
        )

    arguments = mocked_runner.call_args.kwargs
    train_datasets = arguments["train_datasets"]

    assert result == expected_result
    mocked_build_model.assert_called_once_with(num_classes=4)

    assert arguments["class_splits"] == (
        (4, 6, 1, 7),
        (0, 3),
        (2, 5),
    )

    assert [
        {experience[i][1] for i in range(len(experience))}
        for experience in train_datasets
    ] == [
        {0, 1, 2, 3},
        {4, 5},
        {6, 7},
    ]

    if method in {"prototype", "boundary", "hybrid"}:
        assert arguments["selection_strategy"] == method
