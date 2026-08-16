"""Test configs."""

from ml_switcheroo_compiler.ops.configs import (
    ConvDimensionNumbers,
    ConvGeneralDilatedDimensionNumbers,
    DotDimensionNumbers,
    GatherDimensionNumbers,
    GatherScatterMode,
    Precision,
    PrecisionLike,
    RandomAlgorithm,
    RoundingMethod,
    ScatterDimensionNumbers,
)


def test_configs() -> None:
    """Test configs."""
    ConvDimensionNumbers([1], [1], [1])
    ConvGeneralDilatedDimensionNumbers([1], [1], [1])
    DotDimensionNumbers([1], [1], [1], [1])
    GatherDimensionNumbers([1], [1], [1])
    GatherScatterMode()
    Precision()
    PrecisionLike()
    RandomAlgorithm()
    RoundingMethod()
    ScatterDimensionNumbers([1], [1], [1])


def test_root_registry_config_methods():
    from ml_switcheroo_compiler.ops.config_models import OpRegistryConfig, OpsRegistry

    cfg = OpsRegistry(root={"TestOp": OpRegistryConfig(variants={})})

    assert "TestOp" in cfg.dict()
    items = list(cfg.items())
    assert len(items) == 1
    assert items[0][0] == "TestOp"
    assert cfg.get("TestOp") is not None
    assert cfg.get("MissingOp") is None
