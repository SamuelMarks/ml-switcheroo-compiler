from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.linalg.solvers as mod
from ml_switcheroo_compiler.core.config import config


class DummyTensor:
    def __init__(self, shape):
        self.shape = shape
        self.data = np.zeros(shape)
        self.dtype = "float32"
        self.device = None


@patch("ml_switcheroo_compiler.ops.linalg.solvers._emit_linalg_node")
def test_solvers_tracing(mock_emit):
    config.eager_mode = False

    a = DummyTensor((2, 2))
    b = DummyTensor((2, 2))
    c = DummyTensor((2,))

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.CholeskySolve.infer_shape", return_value=(2, 2)):
        mod.cholesky_solve(a, b)

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.BandedTriangularSolve.infer_shape", return_value=(2, 2)):
        mod.banded_triangular_solve(a, b)

    mod.conjugate_gradient(a, b)

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.Lstsq.infer_shape", return_value=(2, 2)):
        mod.lstsq(a, b)

    mod.lu(a)

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.LuMatrixInverse.infer_shape", return_value=(2, 2)):
        mod.lu_matrix_inverse(a, b)

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.LuReconstruct.infer_shape", return_value=(2, 2)):
        mod.lu_reconstruct(a, b)

    mod.lu_solve(a, b, c)

    with patch("ml_switcheroo_compiler.ops.linalg.decompositions.misc.TriangularSolve.infer_shape", return_value=(2, 2)):
        mod.triangular_solve(a, b)

    mod.tridiagonal_solve(a, b)

    mod.tensorinv(a)
    mod.tensorsolve(a, b)


@patch("ml_switcheroo_compiler.ops.linalg.solvers.get_active_backend")
def test_solvers_eager(mock_get_backend):
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


def test_solvers_opdefs():
    # Lstsq
    op1 = mod.Lstsq()
    assert op1.infer_shape(DummyTensor((2, 2)), DummyTensor((2,))) == (2,)
    assert op1.infer_shape(DummyTensor((2, 2)), DummyTensor((2, 2))) == (2, 2)

    # Pinv
    op2 = mod.Pinv()
    assert op2.infer_shape(DummyTensor((2, 3))) == (3, 2)
    assert op2.infer_shape("not_tensor") == ()

    # Sqrtm
    op3 = mod.Sqrtm()
    assert op3.infer_shape(DummyTensor((2, 2))) == (2, 2)

    # CholeskySolve
    op4 = mod.CholeskySolve()
    assert op4.infer_shape(DummyTensor((2, 2)), DummyTensor((2,))) == (2,)

    # BandedTriangularSolve
    op5 = mod.BandedTriangularSolve()
    assert op5.infer_shape(DummyTensor((2, 2)), DummyTensor((2,))) == (2,)

    # EighTridiagonal
    op6 = mod.EighTridiagonal()
    assert op6.infer_shape(DummyTensor((2,)), DummyTensor((1,))) == ((2,), [2, 2])

    # MatrixNorm
    op7 = mod.MatrixNorm()
    assert op7.infer_shape(DummyTensor((2, 2))) == ()

    # VectorNorm
    op8 = mod.VectorNorm()
    assert op8.infer_shape(DummyTensor((2,))) == ()

    # Svdvals
    op9 = mod.Svdvals()
    assert op9.infer_shape(DummyTensor((2, 2))) == (2,)
    assert op9.infer_shape(DummyTensor((2,))) == ()

    # Tensorinv
    op10 = mod.Tensorinv()
    assert op10.infer_shape(DummyTensor((2, 2))) == (2, 2)

    # Tensorsolve
    op11 = mod.Tensorsolve()
    assert op11.infer_shape(DummyTensor((2, 2, 2)), DummyTensor((2,))) == (2, 2)

    # LuMatrixInverse
    op12 = mod.LuMatrixInverse()
    assert op12.infer_shape(DummyTensor((2, 2)), DummyTensor((2,))) == (2, 2)

    # LuReconstruct
    op13 = mod.LuReconstruct()
    assert op13.infer_shape(DummyTensor((2, 2)), DummyTensor((2,))) == (2, 2)
