import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from incremental_blood_cell.evaluator import evaluate_tasks


def test_evaluate_task() -> None:
    model = nn.Linear(in_features=1, out_features=2, bias=False)

    with torch.no_grad():
        model.weight.copy_(torch.tensor([[-1.0], [1.0]]))

    first_task = TensorDataset(
        torch.tensor([[-2.0], [2.0], [-1.0], [1.0]]), torch.tensor([0, 1, 0, 1])
    )

    second_task = TensorDataset(torch.tensor([[-1.0], [1.0]]), torch.tensor([1, 1]))

    model.train()

    accuracies = evaluate_tasks(
        model=model,
        datasets=(first_task, second_task),
        batch_size=2,
    )

    assert accuracies == pytest.approx((1.0, 0.5))
    assert model.training
