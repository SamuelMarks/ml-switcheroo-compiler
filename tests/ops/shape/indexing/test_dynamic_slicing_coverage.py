"""Tests for test_dynamic_slicing_coverage."""

from __future__ import annotations

import importlib
import typing
from unittest.mock import patch

import numpy as np
import pytest

import ml_switcheroo_compiler.ops.shape.dynamic_slicing as dynamic_slicing_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol

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


def test_shape_dynamic_slicing_coverage() -> None:
    """Verify 100% line and branch coverage for dynamic_slicing operations."""
    orig_eager = config.eager_mode
    try:
        # dynamic_slice eager mode
        config.eager_mode = True
        arr = np.arange(16).reshape(4, 4)
        t_arr = Tensor(arr, TensorConfig((4, 4), DType.Int32, Device("cpu")))
        start_0 = Tensor(np.array(1), TensorConfig((), DType.Int32, Device("cpu")))
        start_1 = 2  # integer without .data

        res = typing.cast("EagerCallableProtocol", dynamic_slicing_mod.dynamic_slice)(t_arr, [start_0, start_1], [2, 2])
        assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)
        assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerTensorProtocol", getattr(res, "data")), typing.cast("EagerCallableProtocol", np.array)([[6, 7], [10, 11]]))

        # dynamic_slice non-eager mode
        config.eager_mode = False
        # Mismatched rank
        with pytest.raises(ValueError, match="slice_sizes length"):
            dynamic_slicing_mod.dynamic_slice(t_arr, [start_0], [2, 2, 2])

        # Valid rank
        with patch("ml_switcheroo_compiler.ops.shape.dynamic_slicing._emit_shape_node", return_value="emitted_dynamic_slice") as mock_emit:
            res_graph = dynamic_slicing_mod.dynamic_slice(t_arr, [start_0, start_0], [2, 2])
            assert res_graph == "emitted_dynamic_slice"
            mock_emit.assert_called_once_with(
                "DynamicSlice",
                [t_arr, start_0, start_0],
                {"slice_sizes": (2, 2)},
                (2, 2),
                t_arr.dtype,
            )

        # update_slice and dynamic_update_slice
        with patch("ml_switcheroo_compiler.ops.shape.dynamic_slicing.dynamic_update_slice", return_value="updated_slice") as mock_update:
            res_up = typing.cast("EagerCallableProtocol", dynamic_slicing_mod.update_slice)(t_arr, t_arr, [start_0, 0])
            assert res_up == "updated_slice"
            mock_update.assert_called_once()

        with patch("ml_switcheroo_compiler.ops.shape.dynamic_slicing._emit_shape_node", return_value="emitted_dynamic_update"):
            res_dus = dynamic_slicing_mod.dynamic_update_slice(t_arr, t_arr, [start_0, start_0])
            assert res_dus == "emitted_dynamic_update"

        # DynamicSlice.infer_shape
        op_slice = dynamic_slicing_mod.DynamicSlice()
        assert op_slice.infer_shape(None, None, (3, 4)) == (3, 4)
        assert op_slice.infer_shape(None, slice_sizes=(5, 6)) == (5, 6)

        # DynamicUpdateSlice.infer_shape
        op_update = dynamic_slicing_mod.DynamicUpdateSlice()
        assert typing.cast("EagerCallableProtocol", op_update.infer_shape)(typing.cast("EagerCallableProtocol", DummyWithShape)((4, 4)), None, None) == (4, 4)
        assert typing.cast("EagerCallableProtocol", op_update.infer_shape)(typing.cast("EagerCallableProtocol", DummyWithoutShape)(), None, None) == ()
    finally:
        config.eager_mode = orig_eager
