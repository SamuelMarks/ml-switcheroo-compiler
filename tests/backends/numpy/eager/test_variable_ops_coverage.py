"""Tests for test_variable_ops_coverage."""

from __future__ import annotations

import types

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import variable_ops as var_module


def test_variable_ops() -> None:
    """Test variable operations directly and through eager registry.

    Returns:
        None
    """
    assert var_module._np_assign(np, 10, 20) == 20
    reg_assign = numpy_eager_registry.get("Assign")
    assert reg_assign(np, 10, 20) == 20

    # Cast branches
    arr = np.array([1, 2], dtype=np.int32)
    # bfloat
    cast_bf = var_module._np_cast(np, arr, "bfloat16")
    assert cast_bf.dtype == np.float32
    # float8
    cast_f8 = var_module._np_cast(np, arr, "float8_e4m3")
    assert cast_f8.dtype == np.float32
    # int4
    cast_i4 = var_module._np_cast(np, arr, "int4")
    assert cast_i4.dtype == np.int8
    # normal string
    cast_f64 = var_module._np_cast(np, arr, "float64")
    assert cast_f64.dtype == np.float64
    # with .value
    dummy_dt = types.SimpleNamespace(value="int64")
    cast_ns = var_module._np_cast(np, arr, dummy_dt)
    assert cast_ns.dtype == np.int64
    # numpy dtype direct
    cast_np_dt = var_module._np_cast(np, arr, np.float32)
    assert cast_np_dt.dtype == np.float32

    reg_cast = numpy_eager_registry.get("Cast")
    assert reg_cast(np, arr, "float64").dtype == np.float64

    # Bitcast
    arr_int32 = np.array([1065353216], dtype=np.int32)
    bitcast_f32 = var_module._np_bitcast(np, arr_int32, np.float32)
    assert bitcast_f32.dtype == np.float32
    bitcast_ns = var_module._np_bitcast(np, arr_int32, types.SimpleNamespace(value=np.float32))
    assert bitcast_ns.dtype == np.float32

    reg_bitcast = numpy_eager_registry.get("Bitcast")
    assert reg_bitcast(np, arr_int32, np.float32).dtype == np.float32

    # ReadVariable
    assert var_module._np_read_variable(np, 42) == 42
    assert var_module._np_read_variable(np) is None
    reg_read = numpy_eager_registry.get("ReadVariable")
    assert reg_read(np, 99) == 99

    # AssignVariable
    assert var_module._np_assign_variable(np, 1, 2) == 2
    assert var_module._np_assign_variable(np, 1) is None
    assert var_module._np_assign_variable(np) is None
    reg_assign_var = numpy_eager_registry.get("AssignVariable")
    assert reg_assign_var(np, 1, 3) == 3
