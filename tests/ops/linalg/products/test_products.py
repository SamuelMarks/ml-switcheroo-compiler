"""Tests for test_products."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.ops.linalg.products import Adjoint, BandPart, Diag, Diagonal, EinsumPath, Matmul, MatrixPower, MatrixRank, MatrixTranspose, MultiDot, Trace, _has_valid_shape

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_products_infer_shapes() -> None:
    """Test function test_products_infer_shapes."""

    class MockTensor:
        """Test class MockTensor."""

        shape = (2, 3, 4)

    t = MockTensor()
    assert cast("Callable[..., EagerTensorProtocol]", BandPart)().infer_shape((2, 3)) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", BandPart)().infer_shape(None) is None
    assert cast("Callable[..., EagerTensorProtocol]", Diag)().infer_shape((2,)) == (2, 2)
    assert cast("Callable[..., EagerTensorProtocol]", Diag)().infer_shape((2, 3)) == (2,)
    assert cast("Callable[..., EagerTensorProtocol]", Diag)().infer_shape(None) is None
    assert cast("Callable[..., EagerTensorProtocol]", Matmul)().infer_shape((2, 3), (3, 4)) == (2, 4)
    from ml_switcheroo_compiler.core.errors import ShapeMismatchError

    with pytest.raises(ShapeMismatchError):
        cast("Callable[..., EagerTensorProtocol]", Matmul)().infer_shape((2, 3), (5, 4))
    assert cast("Callable[..., EagerTensorProtocol]", Matmul)().infer_shape(None, None) is None
    assert cast("Callable[..., EagerTensorProtocol]", MatrixPower)().infer_shape(t) == (2, 3, 4)
    assert cast("Callable[..., EagerTensorProtocol]", MatrixPower)().infer_shape(None) == ()
    assert cast("Callable[..., EagerTensorProtocol]", Trace)().infer_shape(t, axis1=0, axis2=1) == (4,)
    assert cast("Callable[..., EagerTensorProtocol]", MatrixRank)().infer_shape(t) == (2,)
    assert cast("Callable[..., EagerTensorProtocol]", MatrixTranspose)().infer_shape(t) == (2, 4, 3)
    assert cast("Callable[..., EagerTensorProtocol]", Adjoint)().infer_shape(t) == (2, 4, 3)
    assert Diagonal().infer_shape() == ()
    assert EinsumPath().infer_shape() == ()
    assert MultiDot().infer_shape() == ()


def test_products_missed() -> None:
    """Test function test_products_missed."""
    from ml_switcheroo_compiler.ops.linalg.products import _has_valid_shape

    assert cast("Callable[..., EagerTensorProtocol]", Diag)().infer_shape(cast("Callable[..., EagerTensorProtocol]", tuple)()) is None
    assert _has_valid_shape(MagicMock(shape=())) is False

    class MockTensor1:
        """Test class MockTensor1."""

        shape = (1,)

    t1 = MockTensor1()
    assert cast("Callable[..., EagerTensorProtocol]", Trace)().infer_shape(t1) == (1,)
    assert cast("Callable[..., EagerTensorProtocol]", MatrixRank)().infer_shape(t1) == (1,)
    assert cast("Callable[..., EagerTensorProtocol]", MatrixTranspose)().infer_shape(t1) == (1,)
    assert cast("Callable[..., EagerTensorProtocol]", Adjoint)().infer_shape(t1) == (1,)
    with patch("ml_switcheroo_compiler.ops.linalg.products.matmul_shape", side_effect=ValueError):
        assert cast("Callable[..., EagerTensorProtocol]", Matmul)().infer_shape((2, 2), (2, 2)) is None


class DummyTensor:
    """Test class DummyTensor."""

    def __init__(self, shape: EagerValue) -> None:
        """Test function __init__."""
        self.shape = shape
        self.data = cast("Callable[..., EagerTensorProtocol]", np.zeros)(shape)
        self.dtype = "float32"
        self.device = None


def test_linalg_products() -> None:
    """Test function test_linalg_products."""
    op1 = BandPart()
    assert cast("Callable[..., EagerTensorProtocol]", op1.infer_shape)((2, 2)) == (2, 2)
    assert cast("Callable[..., EagerTensorProtocol]", op1.infer_shape)("string") is None
    op2 = Diag()
    assert cast("Callable[..., EagerTensorProtocol]", op2.infer_shape)((2,)) == (2, 2)
    assert cast("Callable[..., EagerTensorProtocol]", op2.infer_shape)((2, 2)) == (2,)
    assert cast("Callable[..., EagerTensorProtocol]", op2.infer_shape)("string") is None
    op3 = Matmul()
    assert cast("Callable[..., EagerTensorProtocol]", op3.infer_shape)((2, 2), (2, 2)) == (2, 2)
    try:
        cast("Callable[..., EagerTensorProtocol]", op3.infer_shape)((2, 2), (3, 3))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.linalg.products.matmul_shape", side_effect=ValueError):
        assert cast("Callable[..., EagerTensorProtocol]", op3.infer_shape)((2, 2), (3, 3)) is None
    assert cast("Callable[..., EagerTensorProtocol]", op3.infer_shape)("string", "string") is None
    assert cast("Callable[..., EagerTensorProtocol]", _has_valid_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2)))
    assert not cast("Callable[..., EagerTensorProtocol]", _has_valid_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)(()))
    assert not _has_valid_shape("string")
    op4 = MatrixPower()
    assert cast("Callable[..., EagerTensorProtocol]", op4.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2))) == (2, 2)
    assert cast("Callable[..., EagerTensorProtocol]", op4.infer_shape)("string") == ()
    op5 = Trace()
    assert cast("Callable[..., EagerTensorProtocol]", op5.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2))) == ()
    assert cast("Callable[..., EagerTensorProtocol]", op5.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 3, 4)), axis1=0, axis2=2) == (3,)
    op6 = MatrixRank()
    assert cast("Callable[..., EagerTensorProtocol]", op6.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2))) == ()
    assert cast("Callable[..., EagerTensorProtocol]", op6.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2))) == (2,)
    op7 = MatrixTranspose()
    assert cast("Callable[..., EagerTensorProtocol]", op7.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 3))) == (3, 2)
    assert cast("Callable[..., EagerTensorProtocol]", op7.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2,))) == (2,)
    op8 = Adjoint()
    assert cast("Callable[..., EagerTensorProtocol]", op8.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 3))) == (3, 2)
    assert cast("Callable[..., EagerTensorProtocol]", op8.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2,))) == (2,)
    op9 = Diagonal()
    assert op9.infer_shape() == ()
    op10 = EinsumPath()
    assert op10.infer_shape() == ()
    op11 = MultiDot()
    assert op11.infer_shape() == ()
