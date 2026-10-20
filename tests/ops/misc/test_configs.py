"""Test configs."""

from __future__ import annotations

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
    ConvDimensionNumbers(lhs_spec=[1], rhs_spec=[1], out_spec=[1])
    ConvGeneralDilatedDimensionNumbers(lhs_spec=[1], rhs_spec=[1], out_spec=[1])
    DotDimensionNumbers(lhs_contracting_dimensions=[1], rhs_contracting_dimensions=[1], lhs_batch_dimensions=[1], rhs_batch_dimensions=[1])
    GatherDimensionNumbers(offset_dims=[1], collapsed_slice_dims=[1], start_index_map=[1])
    GatherScatterMode()
    Precision()
    PrecisionLike()
    RandomAlgorithm()
    RoundingMethod()
    ScatterDimensionNumbers(update_window_dims=[1], inserted_window_dims=[1], scatter_dims_to_operand_dims=[1])
    assert callable(ScatterDimensionNumbers)


def test_root_registry_config_methods() -> None:
    """Test function test_root_registry_config_methods."""
    from ml_switcheroo_compiler.ops.config_models import OpRegistryConfig, OpsRegistry

    cfg = OpsRegistry(root={"TestOp": OpRegistryConfig(variants={})})

    assert "TestOp" in cfg.model_dump()
    items = list(cfg.items())
    assert len(items) == 1
    assert items[0][0] == "TestOp"
    assert cfg.get("TestOp") is not None
    assert cfg.get("MissingOp") is None
