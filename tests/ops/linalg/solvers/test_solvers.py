"""Tests for consolidated linear solver operations (`ml_switcheroo_compiler.ops.linalg.solvers`)."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.linalg.solvers as mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.linalg.decompositions.inv import Pinv
from ml_switcheroo_compiler.ops.linalg.decompositions.norms import MatrixNorm, Svdvals, VectorNorm
from ml_switcheroo_compiler.ops.linalg.products import EighTridiagonal, Sqrtm
from tests.testing_utils import EagerTensorProtocol, MockTensorStub, coerce_int_list

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg, EagerTensorProtocol


class DummyTensor(MockTensorStub):
    """Lightweight mock tensor with shape, data, dtype, and device attributes."""

    def __init__(self, shape: BackendArg) -> None:
        """Initialize dummy tensor.

        Args:
            shape: Shape tuple for the mock tensor.
        """
        self.shape = shape
        self.data = np.zeros(coerce_int_list(shape))
        self.dtype = "float32"
        self.device = None


@patch("ml_switcheroo_compiler.ops.linalg.solvers._emit_linalg_node")
def test_solvers_tracing(mock_emit: EagerTensorProtocol) -> None:
    """Verify symbolic graph emission across all consolidated solver functions.

    Args:
        mock_emit: Mocked `_emit_linalg_node` callable.
    """
    config.eager_mode = False

    a = DummyTensor((2, 2))
    b = DummyTensor((2, 2))
    c = DummyTensor((2,))

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.CholeskySolve.infer_shape", return_value=(2, 2)):
        mod.cholesky_solve(a, b)

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.BandedTriangularSolve.infer_shape", return_value=(2, 2)):
        mod.banded_triangular_solve(a, b)

    assert mod.conjugate_gradient(a, b) is b

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.Lstsq.infer_shape", return_value=(2, 2)):
        mod.lstsq(a, b)

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.LuMatrixInverse.infer_shape", return_value=(2, 2)):
        mod.lu_matrix_inverse(a, b)

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.LuReconstruct.infer_shape", return_value=(2, 2)):
        mod.lu_reconstruct(a, b)

    mod.lu_solve((a, b), c)

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.TriangularSolve.infer_shape", return_value=(2, 2)):
        mod.triangular_solve(a, b)

    mod.tridiagonal_solve(c, c, c, c)

    mod.tensorinv(a)
    mod.tensorsolve(a, b)
    assert c.shape == (2,)


@patch("ml_switcheroo_compiler.ops.linalg.solvers.get_active_backend")
def test_solvers_eager(mock_get_backend: EagerTensorProtocol) -> None:
    """Verify eager backend dispatch across consolidated solver functions.

    Args:
        mock_get_backend: Mocked `get_active_backend` callable.
    """
    config.eager_mode = True

    mock_backend = MagicMock()
    mock_backend.execute_op.return_value = np.zeros((2, 2))
    mock_backend.array.return_value = np.zeros((2, 2))
    mock_backend.array.return_value.shape = (2, 2)
    mock_get_backend.return_value = mock_backend

    a = DummyTensor((2, 2))
    b = DummyTensor((2, 2))
    c = DummyTensor((2,))

    mod.cholesky_solve(a, b)
    mod.banded_triangular_solve(a, b)

    mod.lstsq(a, b)
    mock_backend.execute_op.return_value = (np.zeros((2, 2)), np.zeros((2, 2)))
    mod.lstsq(a, b)
    mock_backend.execute_op.return_value = np.zeros((2, 2))

    mod.lu_matrix_inverse(a, b)
    mod.lu_reconstruct(a, b)
    mod.triangular_solve(a, b)
    mod.tensorinv(a)
    mod.tensorsolve(a, b)

    config.eager_mode = False
    assert c.shape == (2,)


def test_solvers_opdefs() -> None:
    """Verify `infer_shape` across solver, inverse, norm, and eigensolver `OpDef` classes."""
    op1 = mod.Lstsq()
    assert op1.infer_shape(DummyTensor((2, 2)), DummyTensor((2,))) == (2,)
    assert op1.infer_shape(DummyTensor((2, 2)), DummyTensor((2, 2))) == (2, 2)

    op2 = Pinv()
    assert op2.infer_shape(DummyTensor((2, 3))) == (3, 2)
    assert op2.infer_shape(DummyTensor((2,))) == (2,)
    assert op2.infer_shape("not_tensor") == ()

    op3 = Sqrtm()
    assert op3.infer_shape(DummyTensor((2, 2))) == (2, 2)

    op4 = mod.CholeskySolve()
    assert op4.infer_shape(DummyTensor((2, 2)), DummyTensor((2,))) == (2,)

    op5 = mod.BandedTriangularSolve()
    assert op5.infer_shape(DummyTensor((2, 2)), DummyTensor((2,))) == (2,)

    op6 = EighTridiagonal()
    assert op6.infer_shape(DummyTensor((2,)), DummyTensor((1,))) == (
        (2,),
        [2, 2],
    )

    op7 = MatrixNorm()
    assert op7.infer_shape(DummyTensor((2, 2))) == ()
    assert op7.infer_shape() == ()

    op8 = VectorNorm()
    assert op8.infer_shape(DummyTensor((2,))) == ()
    assert op8.infer_shape() == ()

    op9 = Svdvals()
    assert op9.infer_shape(DummyTensor((2, 2))) == (2,)
    assert op9.infer_shape(DummyTensor((2,))) == ()

    op10 = mod.Tensorinv()
    assert op10.infer_shape(DummyTensor((2, 2))) == (2, 2)

    op11 = mod.Tensorsolve()
    assert op11.infer_shape(DummyTensor((2, 2, 2)), DummyTensor((2,))) == (2, 2)

    op12 = mod.LuMatrixInverse()
    assert op12.infer_shape(DummyTensor((2, 2)), DummyTensor((2,))) == (2, 2)

    op13 = mod.LuReconstruct()
    assert op13.infer_shape(DummyTensor((2, 2)), DummyTensor((2,))) == (2, 2)
