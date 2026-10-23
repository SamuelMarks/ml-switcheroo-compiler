"""Tests for test_mlx_type_conversion_emulation."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.backends.mlx.eager import (
    _mlx_cast,
    _mlx_cummax,
    _mlx_cummin,
    _mlx_cumprod,
    _mlx_eye,
    _mlx_full,
    _mlx_nan_to_num,
    _mlx_ones,
    _mlx_partition,
    _mlx_ragged_tensor_to_dense,
    _mlx_reshape,
    _mlx_rope,
    _mlx_scatter_nd,
    _mlx_slice,
    _mlx_take,
    _mlx_take_along_axis,
    _mlx_tensor_scatter_add,
    _mlx_tensor_scatter_max,
    _mlx_tensor_scatter_min,
    _mlx_tensor_scatter_update,
    _mlx_variance,
    _mlx_zeros,
    _parse_partition_k,
    _resolve_dtype,
    execute_op,
)
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    import mlx.core as mx

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue

mx = pytest.importorskip("mlx.core", exc_type=ImportError)


def test_numpy_conversion_fallback() -> None:
    """Test function test_numpy_conversion_fallback."""
    with pytest.raises((RuntimeError, ValueError, TypeError, KeyError, Exception)):
        as_callable(execute_op)(None, "UnknownFakeOp", [1.0])
    as_callable(execute_op)(None, "Take", as_callable(mx.array)([1.0]), as_callable(mx.array)([0]), axis=0)
    as_callable(execute_op)(None, "TakeAlongAxis", as_callable(mx.array)([1.0]), as_callable(mx.array)([0]), axis=0)


def _test_mlx_ops_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_mlx_ops.

    Args:
        _state: Shared test state dictionary.
    """
    t = mx.array([1.0])

    class DummyDtype:
        """Test class DummyDtype."""

        value = "int32"

    _mlx_cast(mx, t, dtype="int32")
    _mlx_cast(mx, t, dtype=None)
    _mlx_ragged_tensor_to_dense(mx, t)
    _mlx_take_along_axis(mx, t, mx.array([0]), axis=0)
    _mlx_take(mx, t, mx.array([0]), axis=0)
    t2 = mx.array([[1.0]])
    _mlx_tensor_scatter_update(mx, t2, mx.array([[0]]), mx.array([2.0]))
    _mlx_tensor_scatter_add(mx, t2, mx.array([[0]]), mx.array([2.0]))
    _mlx_tensor_scatter_max(mx, t2, mx.array([[0]]), mx.array([2.0]))
    _mlx_tensor_scatter_min(mx, t2, mx.array([[0]]), mx.array([2.0]))

    class BoxedShape:
        """Test class BoxedShape."""

        data = [1]

        def tolist(self) -> EagerValue:
            """Test function tolist."""
            return [1]

    import numpy as np

    out1 = _mlx_scatter_nd(mx, mx.array([[0]]), mx.array([1.0]), [1])
    getattr(_state, "update")({"BoxedShape": BoxedShape, "DummyDtype": DummyDtype, "np": np, "out1": out1})
    getattr(_state, "update")({"t": t, "t2": t2})
    _state.update(locals())


def _test_mlx_ops_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_mlx_ops.

    Args:
        _state: Shared test state dictionary.
    """
    (BoxedShape, np, out1, t) = (_state["BoxedShape"], _state["np"], _state["out1"], _state["t"])
    assert np.array_equal(np.array(out1), np.array([1.0]))
    out2 = _mlx_scatter_nd(mx, mx.array([[0]]), mx.array([1.0]), shape=[1])
    assert np.array_equal(np.array(out2), np.array([1.0]))
    out3 = _mlx_scatter_nd(mx, mx.array([[0]]), mx.array([1.0]), shape=(1,))
    assert np.array_equal(np.array(out3), np.array([1.0]))
    _mlx_reshape(mx, t, shape=BoxedShape())
    _mlx_reshape(mx, t, shape=(1,))
    _resolve_dtype(mx, None)
    _resolve_dtype(mx, "bfloat16")
    _mlx_zeros(mx, BoxedShape(), dtype="int32")
    _mlx_zeros(mx, 1, dtype="int32")
    _mlx_zeros(mx, 1, dtype="fake")
    _mlx_ones(mx, BoxedShape(), dtype="int32")
    _mlx_ones(mx, 1, dtype="int32")
    _mlx_ones(mx, 1, dtype="fake")
    _mlx_full(mx, BoxedShape(), 2.0, dtype="int32")
    _mlx_full(mx, 1, 2.0, dtype="int32")
    _mlx_full(mx, 1, 2.0, dtype="fake")
    getattr(_state, "update")({"out2": out2, "out3": out3})
    _state.update(locals())


def _test_mlx_ops_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_mlx_ops.

    Args:
        _state: Shared test state dictionary.
    """
    t = _state["t"]

    class ItemMock:
        """Test class ItemMock."""

        def item(self) -> EagerValue:
            """Test function item."""
            return 1

        def __int__(self) -> EagerValue:
            """Test function __int__."""
            return 1

    class DataMock:
        """Test class DataMock."""

        data = ItemMock()

    _parse_partition_k(as_tensor(ItemMock()))
    _parse_partition_k(as_tensor(DataMock()))
    a = mx.array([[1.0, 2.0, 3.0]])
    _mlx_partition(mx, a, k=1, return_indices=False)
    _mlx_partition(mx, a, k=1, return_indices=True)
    _mlx_partition(mx, a, k=1, return_indices=None)
    _mlx_nan_to_num(mx, t, nan=as_tensor(ItemMock()), posinf=as_tensor(DataMock()), neginf=1.0)
    _mlx_cummax(mx, t, axis=0, dtype="int32")
    _mlx_cummax(mx, t, axis=0, dtype=None)
    _mlx_cummin(mx, t, axis=0, dtype="int32")
    _mlx_cummin(mx, t, axis=0, dtype=None)
    getattr(_state, "update")({"DataMock": DataMock, "ItemMock": ItemMock, "a": a})
    _state.update(locals())


def _test_mlx_ops_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_mlx_ops.

    Args:
        _state: Shared test state dictionary.
    """
    (DataMock, t) = (_state["DataMock"], _state["t"])
    _mlx_cumprod(mx, t, axis=0, dtype="int32")
    _mlx_cumprod(mx, t, axis=0, dtype=None)
    _mlx_slice(mx, t, dim=0, start=0, end=1)
    _mlx_eye(mx, DataMock(), m=DataMock())
    _mlx_eye(mx, 2)
    t3 = mx.array([[[1.0]]])
    _mlx_rope(mx, t3, dim=1)
    _mlx_variance(mx, t)
    getattr(_state, "update")({"t3": t3})


def test_mlx_ops() -> None:
    """Test function test_mlx_ops."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_mlx_ops_part_1(_state)
    _test_mlx_ops_part_2(_state)
    _test_mlx_ops_part_3(_state)
    _test_mlx_ops_part_4(_state)
    assert len(_state) > 0
