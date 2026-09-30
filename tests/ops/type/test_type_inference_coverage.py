"""Tests for test_type_inference_coverage."""

from __future__ import annotations

import importlib

import numpy as np

import ml_switcheroo_compiler.ops.type_inference as type_inf_mod
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

gru_mod = importlib.import_module("ml_switcheroo_compiler.ops.nn.gru")


class DummyWithShape:
    """Mock operand providing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock with specific shape.

        Args:
            shape (tuple[int, ...]): Target shape.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking a shape attribute."""

    def __init__(self) -> None:
        """Initialize mock without shape."""
        self.val: int = 42


def test_type_inference_coverage() -> None:
    """Verify 100% line and branch coverage for type_inference operations."""
    dummy_t = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, Device("cpu")))
    dummy_t_int = Tensor(np.array([1]), TensorConfig((1,), DType.Int32, Device("cpu")))

    # 1. res_data has .dtype containing "dtype('int32')"
    class MockDataWithDtypeRegex:
        """Mock result data with dtype repr."""

        dtype: str = "dtype('int32')"

    assert type_inf_mod.resolve_dtype(MockDataWithDtypeRegex(), None) == DType.Int32

    # 2. res_data has .dtype starting with "dtype" but not matching regex
    class MockDataWithDtypePrefix:
        """Mock result data with non-regex dtype starting with dtype."""

        dtype: str = "dtype_custom"

    assert type_inf_mod.resolve_dtype(MockDataWithDtypePrefix(), None) == DType.Float32

    # 3. res_data has .dtype containing dot, e.g. "numpy.float64"
    class MockDataWithDotDtype:
        """Mock result data with dot-separated dtype."""

        dtype: str = "numpy.float64"

    assert type_inf_mod.resolve_dtype(MockDataWithDotDtype(), None) == DType.Float64

    # 4. res_data has .dtype that raises ValueError on DType instantiation
    class MockDataWithInvalidDtype:
        """Mock result data with invalid dtype name."""

        dtype: str = "totally_unknown_dtype_xyz"

    assert type_inf_mod.resolve_dtype(MockDataWithInvalidDtype(), None) == DType.Float32

    # 5. res_data has NO .dtype attribute, first_tensor is provided
    class MockDataNoDtype:
        """Mock result data without dtype attribute."""

    assert type_inf_mod.resolve_dtype(MockDataNoDtype(), dummy_t_int) == DType.Int32

    # 6. res_data has NO .dtype attribute, first_tensor is None
    assert type_inf_mod.resolve_dtype(MockDataNoDtype(), None) == DType.Float32

    # 7. resolve_output_dtype_and_device with "dtype" in kwargs
    out_dt, dev = type_inf_mod.resolve_output_dtype_and_device(dummy_t, {"dtype": DType.Int64})
    assert out_dt == DType.Int64
    assert dev == Device("cpu")

    # 8. resolve_output_dtype_and_device without "dtype" in kwargs, first_tensor provided
    out_dt2, dev2 = type_inf_mod.resolve_output_dtype_and_device(dummy_t, {})
    assert out_dt2 == DType.Float32
    assert dev2 == Device("cpu")

    # 9. resolve_output_dtype_and_device without "dtype" in kwargs, first_tensor is None
    out_dt3, dev3 = type_inf_mod.resolve_output_dtype_and_device(None, {})
    assert out_dt3 == DType.Float32
    assert dev3 is None
