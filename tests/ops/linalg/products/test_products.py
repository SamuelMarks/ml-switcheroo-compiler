"""Tests for test_products."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.ops.linalg.products import Adjoint, BandPart, Diag, Diagonal, EinsumPath, Matmul, MatrixPower, MatrixRank, MatrixTranspose, MultiDot, Trace, _has_valid_shape
from tests.testing_utils import ensure_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_products_infer_shapes() -> None:
    """Test function test_products_infer_shapes."""

    class MockTensor:
        """Test class MockTensor."""

        shape = (2, 3, 4)

    t = MockTensor()
    assert ensure_callable(BandPart)().infer_shape((2, 3)) == (2, 3)
    assert ensure_callable(BandPart)().infer_shape(None) is None
    assert ensure_callable(Diag)().infer_shape((2,)) == (2, 2)
    assert ensure_callable(Diag)().infer_shape((2, 3)) == (2,)
    assert ensure_callable(Diag)().infer_shape(None) is None
    assert ensure_callable(Matmul)().infer_shape((2, 3), (3, 4)) == (2, 4)
    from ml_switcheroo_compiler.core.errors import ShapeMismatchError

    with pytest.raises(ShapeMismatchError):
        ensure_callable(Matmul)().infer_shape((2, 3), (5, 4))
    assert ensure_callable(Matmul)().infer_shape(None, None) is None
    assert ensure_callable(MatrixPower)().infer_shape(t) == (2, 3, 4)
    assert ensure_callable(MatrixPower)().infer_shape(None) == ()
    assert ensure_callable(Trace)().infer_shape(t, axis1=0, axis2=1) == (4,)
    assert ensure_callable(MatrixRank)().infer_shape(t) == (2,)
    assert ensure_callable(MatrixTranspose)().infer_shape(t) == (2, 4, 3)
    assert ensure_callable(Adjoint)().infer_shape(t) == (2, 4, 3)
    assert Diagonal().infer_shape() == ()
    assert EinsumPath().infer_shape() == ()
    assert MultiDot().infer_shape() == ()


def test_products_missed() -> None:
    """Test function test_products_missed."""
    from ml_switcheroo_compiler.ops.linalg.products import _has_valid_shape

    assert ensure_callable(Diag)().infer_shape(ensure_callable(tuple)()) is None
    assert _has_valid_shape(MagicMock(shape=())) is False

    class MockTensor1:
        """Test class MockTensor1."""

        shape = (1,)

    t1 = MockTensor1()
    assert ensure_callable(Trace)().infer_shape(t1) == (1,)
    assert ensure_callable(MatrixRank)().infer_shape(t1) == (1,)
    assert ensure_callable(MatrixTranspose)().infer_shape(t1) == (1,)
    assert ensure_callable(Adjoint)().infer_shape(t1) == (1,)
    with patch("ml_switcheroo_compiler.ops.linalg.products.matmul_shape", side_effect=ValueError):
        assert ensure_callable(Matmul)().infer_shape((2, 2), (2, 2)) is None


class DummyTensor:
    """Test class DummyTensor."""

    def __init__(self, shape: EagerValue) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape
        self.data = ensure_callable(np.zeros)(shape)
        self.dtype = "float32"
        self.device = None


def test_linalg_products() -> None:
    """Test function test_linalg_products."""
    op1 = BandPart()
    assert ensure_callable(op1.infer_shape)((2, 2)) == (2, 2)
    assert ensure_callable(op1.infer_shape)("string") is None
    op2 = Diag()
    assert ensure_callable(op2.infer_shape)((2,)) == (2, 2)
    assert ensure_callable(op2.infer_shape)((2, 2)) == (2,)
    assert ensure_callable(op2.infer_shape)("string") is None
    op3 = Matmul()
    assert ensure_callable(op3.infer_shape)((2, 2), (2, 2)) == (2, 2)
    with pytest.raises((RuntimeError, ValueError, TypeError, KeyError, Exception)):
        ensure_callable(op3.infer_shape)((2, 2), (3, 3))
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.linalg.products.matmul_shape", side_effect=ValueError):
        assert ensure_callable(op3.infer_shape)((2, 2), (3, 3)) is None
    assert ensure_callable(op3.infer_shape)("string", "string") is None
    assert ensure_callable(_has_valid_shape)(ensure_callable(DummyTensor)((2, 2)))
    assert not ensure_callable(_has_valid_shape)(ensure_callable(DummyTensor)(()))
    assert not _has_valid_shape("string")
    op4 = MatrixPower()
    assert ensure_callable(op4.infer_shape)(ensure_callable(DummyTensor)((2, 2))) == (2, 2)
    assert ensure_callable(op4.infer_shape)("string") == ()
    op5 = Trace()
    assert ensure_callable(op5.infer_shape)(ensure_callable(DummyTensor)((2, 2))) == ()
    assert ensure_callable(op5.infer_shape)(ensure_callable(DummyTensor)((2, 3, 4)), axis1=0, axis2=2) == (3,)
    op6 = MatrixRank()
    assert ensure_callable(op6.infer_shape)(ensure_callable(DummyTensor)((2, 2))) == ()
    assert ensure_callable(op6.infer_shape)(ensure_callable(DummyTensor)((2, 2, 2))) == (2,)
    op7 = MatrixTranspose()
    assert ensure_callable(op7.infer_shape)(ensure_callable(DummyTensor)((2, 3))) == (3, 2)
    assert ensure_callable(op7.infer_shape)(ensure_callable(DummyTensor)((2,))) == (2,)
    op8 = Adjoint()
    assert ensure_callable(op8.infer_shape)(ensure_callable(DummyTensor)((2, 3))) == (3, 2)
    assert ensure_callable(op8.infer_shape)(ensure_callable(DummyTensor)((2,))) == (2,)
    op9 = Diagonal()
    assert op9.infer_shape() == ()
    op10 = EinsumPath()
    assert op10.infer_shape() == ()
    op11 = MultiDot()
    assert op11.infer_shape() == ()
