"""Behavioral tests for NumPy eager variable and type-casting operations."""

from __future__ import annotations

import types

import numpy as np

from ml_switcheroo_compiler.backends.eager import variable_ops as var_module
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import coerce_int, extract_test_dtype, require_callable


def test_assign_read_and_assign_variable() -> None:
    """Verify `Assign`, `ReadVariable`, and `AssignVariable` handle present and missing arguments."""
    assert coerce_int(var_module._np_assign(np, 10, 20)) == 20
    reg_assign = numpy_eager_registry.get("Assign")
    assert require_callable(coerce_int)(require_callable(reg_assign)(np, 10, 20)) == 20

    assert coerce_int(var_module._np_read_variable(np, 42)) == 42
    assert var_module._np_read_variable(np) is None
    reg_read = numpy_eager_registry.get("ReadVariable")
    assert require_callable(coerce_int)(require_callable(reg_read)(np, 99)) == 99

    assert coerce_int(var_module._np_assign_variable(np, 1, 2)) == 2
    assert var_module._np_assign_variable(np, "dummy_ref", "real_value") == "real_value"
    assert var_module._np_assign_variable(np, 1) is None
    assert var_module._np_assign_variable(np) is None
    reg_assign_var = numpy_eager_registry.get("AssignVariable")
    assert require_callable(coerce_int)(require_callable(reg_assign_var)(np, 1, 3)) == 3


def test_cast_low_precision_and_dtype_wrappers() -> None:
    """Verify `Cast` maps `bfloat16`/`float8` to `float32`, `int4` to `int8`, and unwraps `.value` dtypes."""
    arr = np.array([1, 2], dtype=np.int32)

    cast_bf16 = ensure_ndarray(var_module._np_cast(np, arr, "bfloat16"))
    assert cast_bf16.dtype == np.float32
    np.testing.assert_array_equal(cast_bf16, np.array([1.0, 2.0], dtype=np.float32))

    cast_f8 = ensure_ndarray(var_module._np_cast(np, arr, "float8_e4m3"))
    assert cast_f8.dtype == np.float32

    cast_i4 = ensure_ndarray(var_module._np_cast(np, arr, "int4"))
    assert cast_i4.dtype == np.int8
    np.testing.assert_array_equal(cast_i4, np.array([1, 2], dtype=np.int8))

    cast_f64 = ensure_ndarray(var_module._np_cast(np, arr, "float64"))
    assert cast_f64.dtype == np.float64

    cast_wrapped = ensure_ndarray(var_module._np_cast(np, arr, types.SimpleNamespace(value="int64")))
    assert cast_wrapped.dtype == np.int64

    cast_np_dt = ensure_ndarray(var_module._np_cast(np, arr, np.float32))
    assert cast_np_dt.dtype == np.float32

    reg_cast = numpy_eager_registry.get("Cast")
    assert extract_test_dtype(require_callable(reg_cast)(np, arr, "float64")) == np.float64


def test_bitcast_reinterprets_ieee754_bits() -> None:
    """Verify `Bitcast` reinterprets raw IEEE-754 `int32` bit patterns as `float32`."""
    ieee_one_bits = np.array([1065353216], dtype=np.int32)

    bitcast_direct = ensure_ndarray(var_module._np_bitcast(np, ieee_one_bits, np.float32))
    assert bitcast_direct.dtype == np.float32
    np.testing.assert_array_equal(bitcast_direct, np.array([1.0], dtype=np.float32))

    bitcast_wrapped = ensure_ndarray(var_module._np_bitcast(np, ieee_one_bits, types.SimpleNamespace(value=np.float32)))
    np.testing.assert_array_equal(bitcast_wrapped, np.array([1.0], dtype=np.float32))

    reg_bitcast = numpy_eager_registry.get("Bitcast")
    np.testing.assert_array_equal(
        ensure_ndarray(require_callable(reg_bitcast)(np, ieee_one_bits, np.float32)),
        np.array([1.0], dtype=np.float32),
    )
