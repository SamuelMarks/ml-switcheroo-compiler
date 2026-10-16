"""Tests for test_distributed."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.ops.reductions.distributed import Pmean, Psum

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, shape: EagerValue) -> None:
        """Test function __init__."""
        self.shape = shape


def test_psum_infer_shape() -> None:
    """Test function test_psum_infer_shape."""
    op = Psum()
    t = MockTensor((2, 3))

    # args
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(t) == (2, 3)

    # kwargs
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(x=t) == (2, 3)

    # empty
    assert op.infer_shape() == ()


def test_pmean_infer_shape() -> None:
    """Test function test_pmean_infer_shape."""
    op = Pmean()
    t = MockTensor((2, 3))

    # args
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(t) == (2, 3)

    # kwargs
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(x=t) == (2, 3)

    # empty
    assert op.infer_shape() == ()
