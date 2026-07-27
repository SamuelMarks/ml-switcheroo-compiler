from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.linalg.matrix_ops as mod
from ml_switcheroo_compiler.core.config import config


class DummyTensor:
    def __init__(self, shape):
        self.shape = shape
        self.data = np.zeros(shape)
        self.dtype = "float32"
        self.device = None


@patch("ml_switcheroo_compiler.ops.linalg.matrix_ops._emit_linalg_node")
def test_matrix_ops_tracing(mock_emit):
    config.eager_mode = False

    a = DummyTensor((2, 2))
    b = DummyTensor((2, 2))

    mod.band_part(a, 1, 1)
    mod.diag(a, 0)
    mod.cross(a, b)
    mod.cross(a, b, axes={"axis": 0})

    with patch("ml_switcheroo_compiler.ops.linalg.products.Trace.infer_shape", return_value=(2,)):
        mod.trace(a)

    with patch("ml_switcheroo_compiler.ops.linalg.products.MatrixRank.infer_shape", return_value=(1,)):
        mod.matrix_rank(a)

    with patch("ml_switcheroo_compiler.ops.linalg.products.MatrixTranspose.infer_shape", return_value=(2, 2)):
        mod.matrix_transpose(a)

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.Sqrtm.infer_shape", return_value=(2, 2)):
        try:
            mod.sqrtm(a)
        except Exception:
            pass

    mod.tensor_diag(a)
    mod.tensor_diag_part(a)
    mod.diag_part(a)

    with patch("ml_switcheroo_compiler.ops.linalg.products.Adjoint.infer_shape", return_value=(2, 2)):
        mod.adjoint(a)

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.EighTridiagonal.infer_shape", return_value=(2,)):
        mod.eigh_tridiagonal(a, b)

    try:
        mod.expm(a)
    except NameError:
        pass

    mod.global_norm([a])
    mod.global_norm([])
    mod.logdet(a)
    mod.logm(a)
    mod.normalize(a)
    mod.set_diag(a, b)

    mod.tridiagonal_matmul(a, a, a, a)
    mod.matrix_norm(a)
    mod.vector_norm(a)

    mod.svdvals(a)
    mod.diagonal(a)
    mod.cond(a)


@patch("ml_switcheroo_compiler.ops.linalg.matrix_ops.get_active_backend")
def test_matrix_ops_eager(mock_get_backend):
    config.eager_mode = True

    mock_backend = MagicMock()
    mock_backend.execute_op.return_value = np.zeros((2, 2))
    mock_backend.array.return_value = np.zeros((2, 2))
    mock_backend.array.return_value.shape = (2, 2)
    mock_get_backend.return_value = mock_backend

    a = DummyTensor((2, 2))
    b = DummyTensor((2, 2))

    mod.band_part(a, 1, 1)
    mod.diag(a, 0)
    mod.cross(a, b)
    mod.trace(a)
    mod.matrix_rank(a)
    mod.matrix_transpose(a)
    try:
        mod.sqrtm(a)
    except Exception:
        pass
    mod.tensor_diag(a)
    mod.tensor_diag_part(a)
    mod.diag_part(a)
    mod.adjoint(a)
    mod.eigh_tridiagonal(a, b)

    mod.tridiagonal_matmul(a, a, a, a)
    mod.matrix_norm(a)
    mod.vector_norm(a)

    mod.svdvals(a)
    mod.diagonal(a)
    mod.cond(a)

    config.eager_mode = False


def test_matrix_ops_opdefs():
    # Svdvals
    op1 = mod.Svdvals()
    assert op1.infer_shape(DummyTensor((2, 3))) == (2,)
    assert op1.infer_shape(DummyTensor((3,))) == (3,)

    # TridiagonalMatmul
    op2 = mod.TridiagonalMatmul()
    assert op2.infer_shape(None, None, None, DummyTensor((2, 2))) == (2, 2)

    # Cond
    op3 = mod.Cond()
    assert op3.infer_shape() == ()
