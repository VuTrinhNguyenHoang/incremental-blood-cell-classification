from incremental_blood_cell.config import ExperimentConfig


def test_builds_configured_class_splits() -> None:
    config = ExperimentConfig(
        method="hybrid",
        class_order=(4, 6, 1, 7, 0, 3, 2, 5),
        seed=17,
    )

    assert config.class_splits == (
        (4, 6, 1, 7),
        (0, 3),
        (2, 5),
    )

    assert config.memory_size == 160
    assert config.distillation_weight == 1.0
    assert config.temperature == 2.0
