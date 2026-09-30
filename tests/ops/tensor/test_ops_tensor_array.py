"""Tests for tensor array operation shape inference."""

from __future__ import annotations

from typing import Union

from ml_switcheroo_compiler.ops.tensor_array import TensorArrayRead, TensorArrayStack, TensorArrayWrite


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

    assert TensorArrayRead().infer_shape(h1, 0) == (2, 3)
    assert TensorArrayRead().infer_shape(h3, 0) == ()
    assert TensorArrayWrite().infer_shape(h1, 0, None) == ()  # type: ignore[arg-type]
    assert TensorArrayStack().infer_shape(h1) == (4, 2, 3)
    assert TensorArrayStack().infer_shape(h2) == (None, 2, 3)
    assert TensorArrayStack().infer_shape(h3) == (None,)
