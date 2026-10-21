"""Tests for test_matrix_ops."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.linalg.matrix_ops as mod
from ml_switcheroo_compiler.core.config import config
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class DummyTensor:
    """Test class DummyTensor."""

    def __init__(self, shape: EagerValue) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape
        self.data = as_callable(np.zeros)(shape)
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
    from ml_switcheroo_compiler.ir.core import IRGraph

    global_tracing_state.active_graph = IRGraph()
    a = DummyTensor((2, 2))
    b = DummyTensor((2, 2))
    as_callable(mod.band_part)(a, 1, 1)
    as_callable(mod.diag)(a, 0)
    as_callable(mod.cross)(a, b)
    as_callable(mod.cross)(a, b, axes={"axis": 0})
    with patch("ml_switcheroo_compiler.ops.linalg.products.Trace.infer_shape", return_value=(2,)):
        as_callable(mod.trace)(a)
    with patch("ml_switcheroo_compiler.ops.linalg.products.MatrixRank.infer_shape", return_value=(1,)):
        as_callable(mod.matrix_rank)(a)
    with patch("ml_switcheroo_compiler.ops.linalg.products.MatrixTranspose.infer_shape", return_value=(2, 2)):
        as_callable(mod.matrix_transpose)(a)
    with patch("ml_switcheroo_compiler.ops.linalg.solvers.Sqrtm.infer_shape", return_value=(2, 2)):
        as_callable(mod.sqrtm)(a)
    as_callable(mod.tensor_diag)(a)
    as_callable(mod.tensor_diag_part)(a)
    as_callable(mod.diag_part)(a)
    with patch("ml_switcheroo_compiler.ops.linalg.products.Adjoint.infer_shape", return_value=(2, 2)):
        as_callable(mod.adjoint)(a)
    with patch("ml_switcheroo_compiler.ops.linalg.solvers.EighTridiagonal.infer_shape", return_value=(2,)):
        as_callable(mod.eigh_tridiagonal)(a, b)
    as_callable(mod.expm)(a)
    as_callable(mod.global_norm)([a])
    mod.global_norm([])
    as_callable(mod.logdet)(a)
    as_callable(mod.logm)(a)
    as_callable(mod.normalize)(a)
    as_callable(mod.set_diag)(a, b)
    as_callable(mod.tridiagonal_matmul)(a, a, a, a)
    as_callable(mod.matrix_norm)(a)
    as_callable(mod.vector_norm)(a)
    as_callable(mod.svdvals)(a)
    as_callable(mod.diagonal)(a)
    as_callable(mod.cond)(a)
    assert callable(test_matrix_ops_tracing)


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
    setattr(mock_get_backend, "return_value", mock_backend)
    a = DummyTensor((2, 2))
    b = DummyTensor((2, 2))
    as_callable(mod.band_part)(a, 1, 1)
    as_callable(mod.diag)(a, 0)
    as_callable(mod.cross)(a, b)
    as_callable(mod.trace)(a)
    as_callable(mod.matrix_rank)(a)
    as_callable(mod.matrix_transpose)(a)
    as_callable(mod.sqrtm)(a)
    as_callable(mod.tensor_diag)(a)
    as_callable(mod.tensor_diag_part)(a)
    as_callable(mod.diag_part)(a)
    as_callable(mod.adjoint)(a)
    as_callable(mod.eigh_tridiagonal)(a, b)
    as_callable(mod.tridiagonal_matmul)(a, a, a, a)
    as_callable(mod.matrix_norm)(a)
    as_callable(mod.vector_norm)(a)
    as_callable(mod.svdvals)(a)
    as_callable(mod.diagonal)(a)
    as_callable(mod.cond)(a)
    config.eager_mode = False
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    global_tracing_state.is_tracing = True
    from ml_switcheroo_compiler.ir.core import IRGraph

    global_tracing_state.active_graph = IRGraph()
    assert callable(test_matrix_ops_eager)


def test_matrix_ops_opdefs() -> None:
    """Test function test_matrix_ops_opdefs."""
    op1 = mod.Svdvals()
    assert as_callable(op1.infer_shape)(as_callable(DummyTensor)((2, 3))) == (2,)
    assert as_callable(op1.infer_shape)(as_callable(DummyTensor)((3,))) == (3,)
    op2 = mod.TridiagonalMatmul()
    assert as_callable(op2.infer_shape)(None, None, None, as_callable(DummyTensor)((2, 2))) == (2, 2)
    op3 = mod.Cond()
    assert op3.infer_shape() == ()
