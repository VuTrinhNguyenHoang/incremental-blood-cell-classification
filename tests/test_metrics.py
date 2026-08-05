import pytest

from incremental_blood_cell.metrics import (
    average_forgetting,
    backward_transfer,
    final_average_accuracy,
)

ACCURACY_MATRIX = (
    (0.90,),
    (0.94, 0.85),
    (0.70, 0.80, 0.88),
)


def test_final_average_accuracy() -> None:
    expected = (0.70 + 0.80 + 0.88) / 3

    assert final_average_accuracy(ACCURACY_MATRIX) == pytest.approx(expected)


def test_average_forgetting() -> None:
    expected = ((0.94 - 0.70) + (0.85 - 0.80)) / 2

    assert average_forgetting(ACCURACY_MATRIX) == pytest.approx(expected)


def test_backward_transfer() -> None:
    expected = ((0.70 - 0.90) + (0.80 - 0.85)) / 2

    assert backward_transfer(ACCURACY_MATRIX) == pytest.approx(expected)


def test_no_forgetting_before_first_increment() -> None:
    matrix = ((0.90,),)

    assert average_forgetting(matrix) == 0.0
    assert backward_transfer(matrix) == 0.0
