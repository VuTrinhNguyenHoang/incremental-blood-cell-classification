from collections.abc import Sequence
from statistics import fmean

AccuracyMatrix = Sequence[Sequence[float]]


def final_average_accuracy(matrix: AccuracyMatrix) -> float:
    return fmean(matrix[-1])


def average_forgetting(matrix: AccuracyMatrix) -> float:
    final_accuracies = matrix[-1]

    if len(final_accuracies) == 1:
        return 0.0

    forgetting = []

    for task_index in range(len(final_accuracies) - 1):
        best_previous_accuracy = max(
            row[task_index] for row in matrix[:-1] if task_index < len(row)
        )

        forgetting.append(best_previous_accuracy - final_accuracies[task_index])

    return fmean(forgetting)


def backward_transfer(matrix: AccuracyMatrix) -> float:
    final_accuracies = matrix[-1]

    if len(final_accuracies) == 1:
        return 0.0

    transfer = [
        final_accuracies[task_index] - matrix[task_index][task_index]
        for task_index in range(len(final_accuracies) - 1)
    ]

    return fmean(transfer)
