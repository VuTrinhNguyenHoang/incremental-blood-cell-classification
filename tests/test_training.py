import random

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from incremental_blood_cell.training import set_seed, train


def test_seed_reproduces_random_values() -> None:
    set_seed(17)

    first_python_value = random.random()
    first_tensor = torch.rand(3)

    set_seed(17)

    second_python_value = random.random()
    second_tensor = torch.rand(3)

    assert first_python_value == second_python_value
    assert torch.equal(first_tensor, second_tensor)


def test_training_updates() -> None:
    set_seed(17)

    model = nn.Linear(2, 2)

    dataset = TensorDataset(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]),
        torch.tensor([0, 1, 1, 0]),
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    optimizer = SGD(model.parameters(), lr=0.1)

    old_params = [param.detach().clone() for param in model.parameters()]

    losses = train(model, loader, optimizer, epochs=2, show_progress=False)

    param_changed = any(
        not torch.equal(old, new)
        for old, new in zip(old_params, model.parameters(), strict=True)
    )

    assert len(losses) == 2
    assert all(loss > 0.0 for loss in losses)
    assert param_changed
    assert model.training
