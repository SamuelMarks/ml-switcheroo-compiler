"""Tests for test_variable_ops_coverage."""

from __future__ import annotations

import types
import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import variable_ops as var_module
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    pass


def test_variable_ops() -> None:
    """Test variable operations directly and through eager registry.

    Returns:
        None
    """
    assert var_module._np_assign(ensure_eager_tensor(np), 10, 20) == 20
    reg_assign = numpy_eager_registry.get("Assign")
    assert ensure_callable(reg_assign)(np, 10, 20) == 20

    # Cast branches
    arr = np.array([1, 2], dtype=np.int32)
    # bfloat
    cast_bf = var_module._np_cast(ensure_eager_tensor(np), arr, "bfloat16")
    assert ensure_eager_tensor(ensure_eager_tensor(cast_bf).dtype) == np.float32
    # float8
    cast_f8 = var_module._np_cast(ensure_eager_tensor(np), arr, "float8_e4m3")
    assert ensure_eager_tensor(ensure_eager_tensor(cast_f8).dtype) == np.float32
    # int4
    cast_i4 = var_module._np_cast(ensure_eager_tensor(np), arr, "int4")
    assert ensure_eager_tensor(ensure_eager_tensor(cast_i4).dtype) == np.int8
    # normal string
    cast_f64 = var_module._np_cast(ensure_eager_tensor(np), arr, "float64")
    assert ensure_eager_tensor(ensure_eager_tensor(cast_f64).dtype) == np.float64
    # with .value
    dummy_dt = types.SimpleNamespace(value="int64")
    cast_ns = var_module._np_cast(ensure_eager_tensor(np), arr, dummy_dt)
    assert ensure_eager_tensor(ensure_eager_tensor(cast_ns).dtype) == np.int64
    # numpy dtype direct
    cast_np_dt = var_module._np_cast(ensure_eager_tensor(np), arr, np.float32)
    assert ensure_eager_tensor(ensure_eager_tensor(cast_np_dt).dtype) == np.float32

    reg_cast = numpy_eager_registry.get("Cast")
    assert ensure_eager_tensor(ensure_eager_tensor(ensure_callable(reg_cast)(np, arr, "float64")).dtype) == np.float64

    # Bitcast
    arr_int32 = np.array([1065353216], dtype=np.int32)
    bitcast_f32 = var_module._np_bitcast(ensure_eager_tensor(np), arr_int32, np.float32)
    assert ensure_eager_tensor(ensure_eager_tensor(bitcast_f32).dtype) == np.float32
    bitcast_ns = var_module._np_bitcast(ensure_eager_tensor(np), arr_int32, types.SimpleNamespace(value=np.float32))
    assert ensure_eager_tensor(ensure_eager_tensor(bitcast_ns).dtype) == np.float32

    reg_bitcast = numpy_eager_registry.get("Bitcast")
    assert ensure_eager_tensor(ensure_eager_tensor(ensure_callable(reg_bitcast)(np, arr_int32, np.float32)).dtype) == np.float32

    # ReadVariable
    assert var_module._np_read_variable(ensure_eager_tensor(np), 42) == 42
    assert var_module._np_read_variable(ensure_eager_tensor(np)) is None
    reg_read = numpy_eager_registry.get("ReadVariable")
    assert ensure_callable(reg_read)(np, 99) == 99

    # AssignVariable
    assert var_module._np_assign_variable(ensure_eager_tensor(np), 1, 2) == 2
    assert var_module._np_assign_variable(ensure_eager_tensor(np), 1) is None
    assert var_module._np_assign_variable(ensure_eager_tensor(np)) is None
    reg_assign_var = numpy_eager_registry.get("AssignVariable")
    assert ensure_callable(reg_assign_var)(np, 1, 3) == 3
