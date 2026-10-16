"""Tests for test_special."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
from ml_switcheroo_compiler.ops.unary.special import Lbeta

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, shape: EagerValue = ()) -> None:
        """Test function __init__."""
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_mvlgamma_infer_shape() -> None:
    """Test function test_mvlgamma_infer_shape."""
    op = Lbeta()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3))) == (2,)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockTensor)(())) == ()


def test_missing_classes_infer_shape() -> None:
    """Test function test_missing_classes_infer_shape."""
    from ml_switcheroo_compiler.ops.unary.special import CanCast, Cast, Frexp

    assert Cast().infer_shape("x") == "x"
    assert CanCast().infer_shape() == ()
    assert Frexp().infer_shape("x") == "x"
