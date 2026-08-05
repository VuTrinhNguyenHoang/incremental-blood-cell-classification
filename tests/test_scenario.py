import pytest

from incremental_blood_cell.scenario import build_class_splits


def test_builds_scenario() -> None:
    splits = build_class_splits(class_order=range(8), increments=(4, 2, 2))

    assert splits == ((0, 1, 2, 3), (4, 5), (6, 7))


def test_rejects_incomplete_increments() -> None:
    with pytest.raises(ValueError, match="cover every class"):
        build_class_splits(class_order=range(8), increments=(4, 2))
