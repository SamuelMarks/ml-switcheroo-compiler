"""Behavioral tests for `ops.creation.api_basic` dtype resolution, fallbacks, and IR tracing modes."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import ShapeProtocol
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.creation.api_basic import (
    _create_backend_array,
    _get_dtype_val,
    _infer_dtype,
    _try_create_array,
    array,
    empty_like,
    full_like,
    ones_like,
    zeros_like,
)
from ml_switcheroo_compiler.tracing.tracer import TracerTape
from tests.testing_utils import MockTestStub

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def _side_effect_type_error(obj: BackendArg, dtype: BackendArg = None) -> list[BackendArg]:
    """Raise `TypeError` when an explicit `dtype` keyword is supplied.

    Args:
        obj (BackendArg): Input payload.
        dtype (BackendArg): Requested dtype value.

    Returns:
        list[BackendArg]: Wrapped single-element list when `dtype` is `None`.

    Raises:
        TypeError: Raised when `dtype` is not `None`.
    """
    if dtype is not None:
        raise TypeError("unsupported dtype")
    return [obj]


def _side_effect_value_error(obj: BackendArg, dtype: BackendArg = None) -> list[BackendArg]:
    """Raise `ValueError` when `dtype` is omitted (`None`).

    Args:
        obj (BackendArg): Input payload.
        dtype (BackendArg): Requested dtype value.

    Returns:
        list[BackendArg]: Wrapped `[obj, dtype]` pair when `dtype` is provided.

    Raises:
        ValueError: Raised when `dtype` is `None`.
    """
    if dtype is None:
        raise ValueError("needs dtype")
    return [obj, dtype]


def test_frontend_basic_dtype_inference_and_value_extraction() -> None:
    """Verify `_infer_dtype` and `_get_dtype_val` handle object arrays, custom dtype holders, and strings."""

    class _BareDummy:
        """Opaque Python element forcing `dtype='O'`."""

    arr_obj = np.array([_BareDummy()], dtype="O")
    assert _infer_dtype(arr_obj) == DType.Object

    class _Int32Holder(MockTestStub):
        """Custom object exposing `.dtype = 'int32'`."""

        dtype = "int32"

    assert _infer_dtype(_Int32Holder()) == DType.Int32
    assert _get_dtype_val("float32") == "float32"
    assert _get_dtype_val(DType.Float32) == "float32"


def test_frontend_basic_try_create_array_and_backend_array() -> None:
    """Verify `_try_create_array` fallback paths and `_create_backend_array` NumPy execution."""
    be_type_err = MagicMock()
    be_type_err.array.side_effect = _side_effect_type_error
    assert _try_create_array(be_type_err, [1, 2], "float32") == [[1, 2]]

    be_val_err = MagicMock()
    be_val_err.array.side_effect = _side_effect_value_error
    assert _try_create_array(be_val_err, [1, 2], None) == [[1, 2], "Any"]

    with ConfigContext(backend="numpy", eager_mode=True):
        created = _create_backend_array([1.0, 2.0], DType.Float32)
        assert isinstance(created, ShapeProtocol)
        assert tuple(created.shape) == (2,)


def test_frontend_basic_like_ops_and_symbolic_array_emit_ir_nodes() -> None:
    """Verify `zeros_like`, `ones_like`, `empty_like`, `full_like`, and `array` emit real IR nodes when tracing."""
    cfg = TensorConfig(shape=(2, 3), dtype=DType.Float32, device=Device("cpu"))
    x = Tensor(np.zeros((2, 3), dtype=np.float32), cfg)

    with ConfigContext(eager_mode=False):
        tape = TracerTape()
        traced_graph = tape.start_tracing("creation_like_ops")
        try:
            z = zeros_like(x)
            o = ones_like(x)
            e = empty_like(x)
            f = full_like(x, fill_value=7.0)
            a = array([1.0, 2.0], dtype=DType.Float32)
        finally:
            tape.stop_tracing()

    assert z.shape == (2, 3)
    assert o.shape == (2, 3)
    assert e.shape == (2, 3)
    assert f.shape == (2, 3)
    assert a.shape == (2,) and a.dtype == DType.Float32

    emitted_ops = {node.op_type for node in traced_graph.nodes.values()}
    assert {"Zeros", "ConstantOfShape", "Constant"} <= emitted_ops
    assert len(traced_graph.nodes) >= 5
