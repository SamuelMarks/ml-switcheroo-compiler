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
