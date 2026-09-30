"""Tests for test_frontend_basic_coverage."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.creation.frontend_basic import (
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
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def _side_effect_type_error(obj: Any, dtype: Any = None) -> list[Any]:
    """Simulate backend array error when dtype is provided.

    Args:
        obj (Any): Input object.
        dtype (Any): Desired dtype.

    Returns:
        list[Any]: Nested list wrapping obj.
    """
    if dtype is not None:
        raise TypeError("unsupported dtype")
    return [obj]


def _side_effect_value_error(obj: Any, dtype: Any = None) -> list[Any]:
    """Simulate backend array error when dtype is missing.

    Args:
        obj (Any): Input object.
        dtype (Any): Desired dtype.

    Returns:
        list[Any]: List with obj and dtype.
    """
    if dtype is None:
        raise ValueError("needs dtype")
    return [obj, dtype]


class _DummyInput:
    """Dummy tensor object holding a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize dummy input.

        Args:
            shape (tuple[int, ...]): Tensor shape.
        """
        self.shape = shape


def test_frontend_basic_dtype_object_and_unknown() -> None:
    """Cover _infer_dtype branches for Object and custom types."""
    arr_obj = np.array([object()], dtype=object)
    assert _infer_dtype(arr_obj) == DType.Object

    class DummyObj:
        """Dummy object with custom dtype."""

        dtype = "int32"

    assert _infer_dtype(DummyObj()) == DType.Int32


def test_frontend_basic_get_dtype_val() -> None:
    """Cover _get_dtype_val when dtype does not have value attribute."""
    assert _get_dtype_val("float32") == "float32"
    assert _get_dtype_val(DType.Float32) == "float32"


def test_frontend_basic_try_create_array_fallback_typeerror() -> None:
    """Cover _try_create_array TypeError fallback when dtype is provided."""
    backend = MagicMock()
    backend.array.side_effect = _side_effect_type_error
    res = _try_create_array(backend, [1, 2], "float32")
    assert res == [[1, 2]]


def test_frontend_basic_try_create_array_fallback_valueerror() -> None:
    """Cover _try_create_array ValueError fallback when dtype_val is None."""
    backend = MagicMock()
    backend.array.side_effect = _side_effect_value_error
    res = _try_create_array(backend, [1, 2], None)
    assert res == [[1, 2], "Any"]


def test_frontend_basic_create_backend_array_calls() -> None:
    """Cover _create_backend_array execution with active backend."""
    backend = MagicMock()
    backend.array.return_value = np.array([1, 2], dtype=np.float32)
    with patch(
        "ml_switcheroo_compiler.ops.creation.frontend_basic.get_active_backend",
        return_value=backend,
    ):
        res = _create_backend_array([1, 2], DType.Float32)
        assert res is not None


def test_frontend_basic_like_funcs_tracing_branch() -> None:
    """Cover zeros_like, ones_like, empty_like, full_like under tracing mode."""
    cfg = TensorConfig(shape=(2, 3), dtype=DType.Float32, device=Device("cpu"))
    x = Tensor(np.zeros((2, 3), dtype=np.float32), cfg)

    global_tracing_state.start_tracing("test_tracing_like")
    try:
        with patch("ml_switcheroo_compiler.ops.creation.frontend_basic._emit_creation_node") as mock_emit:
            mock_emit.return_value = x
            r1 = zeros_like(x)
            r2 = ones_like(x)
            r3 = empty_like(x)
            r4 = full_like(x, fill_value=7.0)
            assert r1 is x
            assert r2 is x
            assert r3 is x
            assert r4 is x
            assert mock_emit.call_count == 4
    finally:
        global_tracing_state.stop_tracing()


def test_frontend_basic_array_symbolic_branch() -> None:
    """Cover array() when eager_mode is False."""
    orig_eager = config.eager_mode
    config.eager_mode = False
    try:
        with patch(
            "ml_switcheroo_compiler.ops.creation.frontend_basic.TracingNodeBuilder.extract_from_constant",
            return_value=("const_node_1",),
        ):
            res = array([1.0, 2.0], dtype=DType.Float32)
            assert isinstance(res, Tensor)
            assert res.shape == (2,)
            assert res.dtype == DType.Float32
    finally:
        config.eager_mode = orig_eager
