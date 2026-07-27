# ruff: noqa: E501
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.ops.linalg.products import Adjoint, BandPart, Diag, Diagonal, EinsumPath, Matmul, MatrixPower, MatrixRank, MatrixTranspose, MultiDot, Trace, _has_valid_shape


def test_products_infer_shapes() -> None:

    class MockTensor:
        shape = (2, 3, 4)

    t = MockTensor()
    assert BandPart().infer_shape((2, 3)) == (2, 3)
    assert BandPart().infer_shape(None) is None
    assert Diag().infer_shape((2,)) == (2, 2)
    assert Diag().infer_shape((2, 3)) == (2,)
    assert Diag().infer_shape(None) is None
    assert Matmul().infer_shape((2, 3), (3, 4)) == (2, 4)
    from ml_switcheroo_compiler.core.errors import ShapeMismatchError

    with pytest.raises(ShapeMismatchError):
        Matmul().infer_shape((2, 3), (5, 4))
    assert Matmul().infer_shape(None, None) is None
    assert MatrixPower().infer_shape(t) == (2, 3, 4)
    assert MatrixPower().infer_shape(None) == ()
    assert Trace().infer_shape(t, axis1=0, axis2=1) == (4,)
    assert MatrixRank().infer_shape(t) == (2,)
    assert MatrixTranspose().infer_shape(t) == (2, 4, 3)
    assert Adjoint().infer_shape(t) == (2, 4, 3)
    assert Diagonal().infer_shape() == ()
    assert EinsumPath().infer_shape() == ()
    assert MultiDot().infer_shape() == ()


def test_products_missed() -> None:
    from ml_switcheroo_compiler.ops.linalg.products import _has_valid_shape

    assert Diag().infer_shape(tuple()) is None
    assert _has_valid_shape(MagicMock(shape=())) is False

    class MockTensor1:
        shape = (1,)

    t1 = MockTensor1()
    assert Trace().infer_shape(t1) == (1,)
    assert MatrixRank().infer_shape(t1) == (1,)
    assert MatrixTranspose().infer_shape(t1) == (1,)
    assert Adjoint().infer_shape(t1) == (1,)
    with patch("ml_switcheroo_compiler.ops.linalg.products.matmul_shape", side_effect=ValueError):
        assert Matmul().infer_shape((2, 2), (2, 2)) is None


class DummyTensor:
    def __init__(self, shape):
        self.shape = shape
        self.data = np.zeros(shape)
        self.dtype = "float32"
        self.device = None


def test_linalg_products():
    op1 = BandPart()
    assert op1.infer_shape((2, 2)) == (2, 2)
    assert op1.infer_shape("string") is None
    op2 = Diag()
    assert op2.infer_shape((2,)) == (2, 2)
    assert op2.infer_shape((2, 2)) == (2,)
    assert op2.infer_shape("string") is None
    op3 = Matmul()
    assert op3.infer_shape((2, 2), (2, 2)) == (2, 2)
    try:
        op3.infer_shape((2, 2), (3, 3))
    except Exception:
        pass
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.linalg.products.matmul_shape", side_effect=ValueError):
        assert op3.infer_shape((2, 2), (3, 3)) is None
    assert op3.infer_shape("string", "string") is None
    assert _has_valid_shape(DummyTensor((2, 2))) == True
    assert _has_valid_shape(DummyTensor(())) == False
    assert _has_valid_shape("string") == False
    op4 = MatrixPower()
    assert op4.infer_shape(DummyTensor((2, 2))) == (2, 2)
    assert op4.infer_shape("string") == ()
    op5 = Trace()
    assert op5.infer_shape(DummyTensor((2, 2))) == ()
    assert op5.infer_shape(DummyTensor((2, 3, 4)), axis1=0, axis2=2) == (3,)
    op6 = MatrixRank()
    assert op6.infer_shape(DummyTensor((2, 2))) == ()
    assert op6.infer_shape(DummyTensor((2, 2, 2))) == (2,)
    op7 = MatrixTranspose()
    assert op7.infer_shape(DummyTensor((2, 3))) == (3, 2)
    assert op7.infer_shape(DummyTensor((2,))) == (2,)
    op8 = Adjoint()
    assert op8.infer_shape(DummyTensor((2, 3))) == (3, 2)
    assert op8.infer_shape(DummyTensor((2,))) == (2,)
    op9 = Diagonal()
    assert op9.infer_shape() == ()
    op10 = EinsumPath()
    assert op10.infer_shape() == ()
    op11 = MultiDot()
    assert op11.infer_shape() == ()
