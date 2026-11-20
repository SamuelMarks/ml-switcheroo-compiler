"""Tests for test_matrix_ops."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.linalg.matrix_ops as mod
from ml_switcheroo_compiler.core.config import config
from tests.testing_utils import BackendArg, EagerTensorProtocol, MockTensorStub, coerce_int_list


class DummyTensor(MockTensorStub):
    """Test class DummyTensor."""

    def __init__(self, shape: BackendArg) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape
        self.data = np.zeros(coerce_int_list(shape))
        self.dtype = "float32"
        self.device = None


@patch("ml_switcheroo_compiler.ops.linalg.matrix_ops._emit_linalg_node")
def test_matrix_ops_tracing(mock_emit: EagerTensorProtocol) -> None:
    """Test function test_matrix_ops_tracing.

    Args:
        mock_emit: Test parameter `mock_emit`.
    """
    config.eager_mode = False
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    global_tracing_state.is_tracing = True
    from ml_switcheroo_ir import IRGraph

    global_tracing_state.active_graph = IRGraph()
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
    with patch("ml_switcheroo_compiler.ops.linalg.products.Sqrtm.infer_shape", return_value=(2, 2)):
        mod.sqrtm(a)
    mod.tensor_diag(a)
    mod.tensor_diag_part(a)
    mod.diag_part(a)
    with patch("ml_switcheroo_compiler.ops.linalg.products.Adjoint.infer_shape", return_value=(2, 2)):
        mod.adjoint(a)
    with patch("ml_switcheroo_compiler.ops.linalg.products.EighTridiagonal.infer_shape", return_value=(2,)):
        mod.eigh_tridiagonal(a, b)
    mod.expm(a)
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
    assert isinstance(mock_emit, MagicMock) and mock_emit.call_count >= 15
    global_tracing_state.is_tracing = False


@patch("ml_switcheroo_compiler.ops.linalg.matrix_ops.get_active_backend")
def test_matrix_ops_eager(mock_get_backend: EagerTensorProtocol) -> None:
    """Test function test_matrix_ops_eager.

    Args:
        mock_get_backend: Test parameter `mock_get_backend`.
    """
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
    mod.sqrtm(a)
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
    assert mock_backend.execute_op.call_count >= 15
    config.eager_mode = False
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    global_tracing_state.is_tracing = True
    from ml_switcheroo_ir import IRGraph

    global_tracing_state.active_graph = IRGraph()
    global_tracing_state.is_tracing = False


def test_matrix_ops_opdefs() -> None:
    """Test function test_matrix_ops_opdefs."""
    op1 = mod.Svdvals()
    assert op1.infer_shape(DummyTensor((2, 3))) == (2,)
    assert op1.infer_shape(DummyTensor((3,))) == (3,)
    op2 = mod.TridiagonalMatmul()
    assert op2.infer_shape(DummyTensor((1,)), DummyTensor((2,)), DummyTensor((1,)), DummyTensor((2, 2))) == (2, 2)
    op3 = mod.Cond()
    assert op3.infer_shape() == ()
