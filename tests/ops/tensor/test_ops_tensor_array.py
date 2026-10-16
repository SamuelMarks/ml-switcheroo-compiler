"""Tests for tensor array operation shape inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union, cast

from ml_switcheroo_compiler.ops.tensor_array import TensorArrayRead, TensorArrayStack, TensorArrayWrite

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


class MockHandle:
    """Mock handle for tensor array operations."""

    def __init__(
        self,
        elem_shape: Union[tuple[int, ...], list[int], None] = (),
        size: Union[int, str, None] = None,
    ) -> None:
        """Initialize mock tensor array handle.

        Args:
            elem_shape (tuple[int, ...] | list[int] | None): Element shape.
            size (int | str | None): Size of the array.
        """
        self.element_shape = elem_shape
        self.size = size


def test_tensor_array_infer_shape() -> None:
    """Test shape inference for TensorArrayRead, TensorArrayWrite, and TensorArrayStack.

    Returns:
        None.
    """
    h1 = MockHandle((2, 3), size=4)
    h2 = MockHandle((2, 3), size=None)
    h3 = MockHandle(elem_shape=(), size="non_int")

    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", TensorArrayRead)().infer_shape)(h1, 0) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", TensorArrayRead)().infer_shape)(h3, 0) == ()
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", TensorArrayWrite)().infer_shape)(h1, 0, None) == ()
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", TensorArrayStack)().infer_shape)(h1) == (4, 2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", TensorArrayStack)().infer_shape)(h2) == (None, 2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", TensorArrayStack)().infer_shape)(h3) == (None,)
