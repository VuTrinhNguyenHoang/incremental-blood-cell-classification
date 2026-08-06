import torch
from torch import nn
from torch.utils.data import TensorDataset

from incremental_blood_cell.methods.finetuning import run_finetuning


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


def test_runs_sequential_finetuning() -> None:
    model = TinyClassifier()

    first_task = TensorDataset(
        torch.tensor(
            [
                [-2.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ]
        ),
        torch.tensor([0, 0, 1, 1]),
    )

    second_task = TensorDataset(
        torch.tensor(
            [
                [0.0, 1.0],
                [0.0, 2.0],
            ]
        ),
        torch.tensor([2, 2]),
    )

    accuracy_matrix = run_finetuning(
        model=model,
        class_splits=((0, 1), (2,)),
        train_datasets=(first_task, second_task),
        test_datasets=(first_task, second_task),
        device=torch.device("cpu"),
        epochs=1,
        batch_size=2,
        learning_rate=0.1,
        show_progress=False,
    )

    assert len(accuracy_matrix) == 2
    assert len(accuracy_matrix[0]) == 1
    assert len(accuracy_matrix[1]) == 2
    assert model.fc.out_features == 3

    assert all(0.0 <= accuracy <= 1.0 for row in accuracy_matrix for accuracy in row)
