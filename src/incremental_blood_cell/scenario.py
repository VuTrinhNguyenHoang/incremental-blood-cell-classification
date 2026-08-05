from collections.abc import Sequence


def build_class_splits(
    class_order: Sequence[int], increments: Sequence[int]
) -> tuple[tuple[int, ...], ...]:
    if sum(increments) != len(class_order):
        raise ValueError("increments must cover every class")

    splits = []
    start = 0
    for size in increments:
        end = start + size
        splits.append(tuple(class_order[start:end]))
        start = end

    return tuple(splits)
