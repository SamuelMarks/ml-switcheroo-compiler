"""Tests for test_distributed."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.ops.reductions.distributed import Pmean, Psum
from tests.testing_utils import MockTensorStub

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


def test_psum_infer_shape() -> None:
    """Test function test_psum_infer_shape."""
    op = Psum()
    t = MockTensor((2, 3))

    # args
    assert op.infer_shape(t) == (2, 3)

    # kwargs
    assert op.infer_shape(x=t) == (2, 3)

    # empty
    assert op.infer_shape() == ()


def test_pmean_infer_shape() -> None:
    """Test function test_pmean_infer_shape."""
    op = Pmean()
    t = MockTensor((2, 3))

    # args
    assert op.infer_shape(t) == (2, 3)

    # kwargs
    assert op.infer_shape(x=t) == (2, 3)

    # empty
    assert op.infer_shape() == ()
