"""Tests for test_special."""

from __future__ import annotations

import typing

# ruff: noqa: E501
from ml_switcheroo_compiler.ops.unary.special import Lbeta

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerValue


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, shape: EagerValue = ()) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_mvlgamma_infer_shape() -> None:
    """Test function test_mvlgamma_infer_shape."""
    op = Lbeta()
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", MockTensor)((2, 3))) == (2,)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", MockTensor)(())) == ()


def test_missing_classes_infer_shape() -> None:
    """Test function test_missing_classes_infer_shape."""
    from ml_switcheroo_compiler.ops.unary.special import CanCast, Cast, Frexp

    assert Cast().infer_shape("x") == "x"
    assert CanCast().infer_shape() == ()
    assert Frexp().infer_shape("x") == "x"
