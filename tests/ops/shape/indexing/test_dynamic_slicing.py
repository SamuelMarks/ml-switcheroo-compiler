"""Tests for test_dynamic_slicing."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.shape.dynamic_slicing import DynamicSlice, DynamicUpdateSlice, dynamic_slice, dynamic_update_slice, update_slice

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue
    from ml_switcheroo_compiler.core.tensor import Tensor


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, shape: EagerValue) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape


def test_dynamic_slice_eager() -> None:
    """Test function test_dynamic_slice_eager."""
    config.eager_mode = True

    class FakeTensor:
        """Test class FakeTensor."""

        def __init__(self, data: EagerValue) -> None:
            """Test function __init__.

            Args:
                data: Test parameter `data`.
            """
            self.data = np.array(data)
            self.shape = self.data.shape
            self.dtype = "float32"
            self.device = "cpu"

    t = FakeTensor([1, 2, 3, 4, 5])

    class FakeStart:
        """Test class FakeStart."""

        def __init__(self, d: EagerValue) -> None:
            """Test function __init__.

            Args:
                d: Test parameter `d`.
            """
            self.data = d

    res = typing.cast("EagerCallableProtocol", dynamic_slice)(t, [typing.cast("EagerCallableProtocol", FakeStart)(1)], [2])
    assert typing.cast("EagerTensorProtocol", getattr(res, "config")).shape == (2,)
    assert list(typing.cast("EagerTensorProtocol", getattr(res, "data"))) == [2, 3]

    res2 = typing.cast("EagerCallableProtocol", dynamic_slice)(t, [3], [2])
    assert typing.cast("EagerTensorProtocol", getattr(res2, "config")).shape == (2,)
    assert list(typing.cast("EagerTensorProtocol", getattr(res2, "data"))) == [4, 5]

    config.eager_mode = False


def test_update_slice(mocker: EagerTensorProtocol) -> None:
    """Test function test_update_slice.

    Args:
        mocker: Test parameter `mocker`.
    """
    mocker.patch("ml_switcheroo_compiler.ops.shape.dynamic_slicing.dynamic_update_slice", return_value="updated")
    mocker.patch("ml_switcheroo_compiler.ops.creation.frontend_basic.array", side_effect=lambda x: f"array({x})")

    t = MockTensor((5,))
    update = MockTensor((2,))

    res = typing.cast("EagerCallableProtocol", update_slice)(t, update, [1, typing.cast("EagerCallableProtocol", MockTensor)(())])
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
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(t, "update", "starts") == (2, 3)

    assert typing.cast("EagerCallableProtocol", op.infer_shape)("not_tensor", "update", "starts") == ()


def test_dynamic_slice_tracing(mocker: EagerTensorProtocol) -> None:
    """Test function test_dynamic_slice_tracing.

    Args:
        mocker: Test parameter `mocker`.
    """
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.dynamic_slicing._emit_shape_node", return_value="sliced")
    t = MockTensor((5,))
    setattr(t, "dtype", "float32")
    res = typing.cast("EagerCallableProtocol", dynamic_slice)(t, [1], [2])
    assert res == "sliced"


def test_dynamic_update_slice_tracing(mocker: EagerTensorProtocol) -> None:
    """Test function test_dynamic_update_slice_tracing.

    Args:
        mocker: Test parameter `mocker`.
    """
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.dynamic_slicing._emit_shape_node", return_value="updated")
    t = MockTensor((5,))
    setattr(t, "dtype", "float32")
    res = dynamic_update_slice(typing.cast("Tensor", t), typing.cast("Tensor", MockTensor((2,))), typing.cast("list[Tensor]", [1]))
    assert res == "updated"


def test_dynamic_slice_validation_and_shapes(mocker: EagerTensorProtocol) -> None:
    """Verify that dynamic_slice computes exact shapes and raises on rank mismatch.

    Args:
        mocker (EagerTensorProtocol): Pytest mocker fixture.
    """
    import pytest

    captured_shape: dict[str, tuple[int, ...]] = {}

    def mock_emit(op_name: str, inputs: list[EagerValue], attrs: dict[str, EagerValue], shape: tuple[int, ...], dtype: EagerValue) -> str:
        """Test function mock_emit.

        Args:
            op_name: Test parameter `op_name`.
            inputs: Test parameter `inputs`.
            attrs: Test parameter `attrs`.
            shape: Test parameter `shape`.
            dtype: Test parameter `dtype`.
        """
        captured_shape[op_name] = shape
        return op_name

    mocker.patch("ml_switcheroo_compiler.ops.shape.dynamic_slicing._emit_shape_node", side_effect=mock_emit)
    config.eager_mode = False

    t = MockTensor((10, 20))
    setattr(t, "dtype", "float32")
    typing.cast("EagerCallableProtocol", dynamic_slice)(t, [0, 0], [4, 8])
    assert captured_shape["DynamicSlice"] == (4, 8)

    with pytest.raises(ValueError, match="slice_sizes length"):
        typing.cast("EagerCallableProtocol", dynamic_slice)(t, [0], [4])
