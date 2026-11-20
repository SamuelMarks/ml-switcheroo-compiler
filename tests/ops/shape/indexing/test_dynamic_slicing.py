"""Behavioral unit tests for dynamic slicing."""

from __future__ import annotations

import importlib
import typing
from unittest.mock import patch

import numpy as np
import pytest
from pytest_mock import MockerFixture

import ml_switcheroo_compiler.ops.shape.dynamic_slicing as dynamic_slicing_mod
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.dynamic_slicing import (
    DynamicSlice,
    DynamicUpdateSlice,
    dynamic_slice,
    dynamic_update_slice,
    update_slice,
)
from tests.testing_utils import (
    MockTensorStub,
    MockTestStub,
    ensure_tensor,
    extract_test_data,
    extract_test_shape,
    require_callable,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class MockTensor(MockTensorStub):
    """Test class MockTensor."""

    def __init__(self, shape: BackendArg) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape


def test_dynamic_slice_eager() -> None:
    """Test function test_dynamic_slice_eager."""
    config.eager_mode = True

    class FakeTensor(MockTensorStub):
        """Test class FakeTensor."""

        def __init__(self, data: BackendArg) -> None:
            """Test function __init__.

            Args:
                data: Test parameter `data`.
            """
            self.data = np.array(data)
            self.shape = self.data.shape
            self.dtype = "float32"
            self.device = "cpu"

    t = FakeTensor([1, 2, 3, 4, 5])

    class FakeStart(MockTensorStub):
        """Test class FakeStart."""

        def __init__(self, d: BackendArg) -> None:
            """Test function __init__.

            Args:
                d: Test parameter `d`.
            """
            self.data = d

    res = dynamic_slice(t, [FakeStart(1)], [2])
    assert ensure_tensor(res).config.shape == (2,)
    assert ensure_ndarray(extract_test_data(res)).tolist() == [2, 3]

    res2 = require_callable(dynamic_slice)(t, [3], [2])
    assert ensure_tensor(res2).config.shape == (2,)
    assert ensure_ndarray(extract_test_data(res2)).tolist() == [4, 5]

    config.eager_mode = False


def test_update_slice(mocker: MockerFixture) -> None:
    """Test function test_update_slice.

    Args:
        mocker: Test parameter `mocker`.
    """
    mocker.patch("ml_switcheroo_compiler.ops.shape.dynamic_slicing.dynamic_update_slice", return_value="updated")
    mocker.patch("ml_switcheroo_compiler.ops.creation.api_basic.array", side_effect=lambda x: f"array({x})")

    t = MockTensor((5,))
    update = MockTensor((2,))

    res = update_slice(t, update, [1, MockTensor(())])
    assert res == "updated"


def test_dynamic_slice_infer_shape() -> None:
    """Test function test_dynamic_slice_infer_shape."""
    op = DynamicSlice()

    # Args
    assert op.infer_shape("x", "starts", [2, 3]) == (2, 3)

    # Kwargs
    assert op.infer_shape(slice_sizes=[4, 5]) == (4, 5)


def test_dynamic_update_slice_infer_shape() -> None:
    """Test function test_dynamic_update_slice_infer_shape."""
    op = DynamicUpdateSlice()

    t = MockTensor((2, 3))
    assert op.infer_shape(t, "update", "starts") == (2, 3)

    assert op.infer_shape("not_tensor", "update", "starts") == ()


def test_dynamic_slice_tracing(mocker: MockerFixture) -> None:
    """Test function test_dynamic_slice_tracing.

    Args:
        mocker: Test parameter `mocker`.
    """
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.dynamic_slicing._emit_shape_node", return_value="sliced")
    t = MockTensor((5,))
    pytest.MonkeyPatch().setattr(t, "dtype", "float32", raising=False)
    res = dynamic_slice(t, [ensure_tensor(1)], [2])
    assert res == "sliced"


def test_dynamic_update_slice_tracing(mocker: MockerFixture) -> None:
    """Test function test_dynamic_update_slice_tracing.

    Args:
        mocker: Test parameter `mocker`.
    """
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.dynamic_slicing._emit_shape_node", return_value="updated")
    t = MockTensor((5,))
    pytest.MonkeyPatch().setattr(t, "dtype", "float32", raising=False)
    res = dynamic_update_slice(ensure_tensor(t), ensure_tensor(MockTensor((2,))), [ensure_tensor(1)])
    assert res == "updated"


def test_dynamic_slice_validation_and_shapes(mocker: MockerFixture) -> None:
    """Verify that dynamic_slice computes exact shapes and raises on rank mismatch.

    Args:
        mocker (MockerFixture): Pytest mocker fixture.
    """
    import pytest

    captured_shape: dict[str, tuple[int, ...]] = {}

    def mock_emit(
        op_name: str, inputs: list[BackendArg], attrs: dict[str, BackendArg], shape: tuple[int, ...], dtype: BackendArg
    ) -> str:
        """Test function mock_emit.

        Args:
            op_name: Test parameter `op_name`.
            inputs: Test parameter `inputs`.
            attrs: Test parameter `attrs`.
            shape: Test parameter `shape`.
            dtype: Test parameter `dtype`.

        Returns:
            str: Computed output value from `mock_emit`.
        """
        captured_shape[op_name] = shape
        return op_name

    mocker.patch("ml_switcheroo_compiler.ops.shape.dynamic_slicing._emit_shape_node", side_effect=mock_emit)
    config.eager_mode = False

    t = MockTensor((10, 20))
    pytest.MonkeyPatch().setattr(t, "dtype", "float32", raising=False)
    dynamic_slice(t, [ensure_tensor(0), ensure_tensor(0)], [4, 8])
    assert captured_shape["DynamicSlice"] == (4, 8)

    with pytest.raises(ValueError, match="slice_sizes length"):
        require_callable(dynamic_slice)(t, [0], [4])


gru_mod = importlib.import_module("ml_switcheroo_compiler.ops.nn.gru")


class DummyWithShape(MockTestStub):
    """Mock operand providing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock with specific shape.

        Args:
            shape (tuple[int, ...]): Target shape.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape(MockTestStub):
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

        res = require_callable(dynamic_slicing_mod.dynamic_slice)(t_arr, [start_0, start_1], [2, 2])
        assert extract_test_shape(res) == (2, 2)
        assert np.array_equal(ensure_ndarray(extract_test_data(res)), np.array([[6, 7], [10, 11]]))

        # dynamic_slice non-eager mode
        config.eager_mode = False
        # Mismatched rank
        with pytest.raises(ValueError, match="slice_sizes length"):
            dynamic_slicing_mod.dynamic_slice(t_arr, [start_0], [2, 2, 2])

        # Valid rank
        with patch(
            "ml_switcheroo_compiler.ops.shape.dynamic_slicing._emit_shape_node", return_value="emitted_dynamic_slice"
        ) as mock_emit:
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
        with patch(
            "ml_switcheroo_compiler.ops.shape.dynamic_slicing.dynamic_update_slice", return_value="updated_slice"
        ) as mock_update:
            res_up = dynamic_slicing_mod.update_slice(t_arr, t_arr, [start_0, 0])
            assert res_up == "updated_slice"
            mock_update.assert_called_once()

        with patch(
            "ml_switcheroo_compiler.ops.shape.dynamic_slicing._emit_shape_node", return_value="emitted_dynamic_update"
        ):
            res_dus = dynamic_slicing_mod.dynamic_update_slice(t_arr, t_arr, [start_0, start_0])
            assert res_dus == "emitted_dynamic_update"

        # DynamicSlice.infer_shape
        op_slice = dynamic_slicing_mod.DynamicSlice()
        assert op_slice.infer_shape(None, None, (3, 4)) == (3, 4)
        assert op_slice.infer_shape(None, slice_sizes=(5, 6)) == (5, 6)

        # DynamicUpdateSlice.infer_shape
        op_update = dynamic_slicing_mod.DynamicUpdateSlice()
        assert op_update.infer_shape(DummyWithShape((4, 4)), None, None) == (4, 4)
        assert op_update.infer_shape(DummyWithoutShape(), None, None) == ()
    finally:
        config.eager_mode = orig_eager
