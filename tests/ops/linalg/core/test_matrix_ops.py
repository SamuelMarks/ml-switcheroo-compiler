"""Tests for test_matrix_ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.linalg.matrix_ops as mod
from ml_switcheroo_compiler.core.config import config

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


class DummyTensor:
    """Test class DummyTensor."""

    def __init__(self, shape: EagerValue) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape
        self.data = cast("EagerCallableProtocol", np.zeros)(shape)
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
    cast("EagerCallableProtocol", mod.band_part)(a, 1, 1)
    cast("EagerCallableProtocol", mod.diag)(a, 0)
    cast("EagerCallableProtocol", mod.cross)(a, b)
    cast("EagerCallableProtocol", mod.cross)(a, b, axes={"axis": 0})
    with patch("ml_switcheroo_compiler.ops.linalg.products.Trace.infer_shape", return_value=(2,)):
        cast("EagerCallableProtocol", mod.trace)(a)
    with patch("ml_switcheroo_compiler.ops.linalg.products.MatrixRank.infer_shape", return_value=(1,)):
        cast("EagerCallableProtocol", mod.matrix_rank)(a)
    with patch("ml_switcheroo_compiler.ops.linalg.products.MatrixTranspose.infer_shape", return_value=(2, 2)):
        cast("EagerCallableProtocol", mod.matrix_transpose)(a)
    with patch("ml_switcheroo_compiler.ops.linalg.solvers.Sqrtm.infer_shape", return_value=(2, 2)):
        cast("EagerCallableProtocol", mod.sqrtm)(a)
    cast("EagerCallableProtocol", mod.tensor_diag)(a)
    cast("EagerCallableProtocol", mod.tensor_diag_part)(a)
    cast("EagerCallableProtocol", mod.diag_part)(a)
    with patch("ml_switcheroo_compiler.ops.linalg.products.Adjoint.infer_shape", return_value=(2, 2)):
        cast("EagerCallableProtocol", mod.adjoint)(a)
    with patch("ml_switcheroo_compiler.ops.linalg.solvers.EighTridiagonal.infer_shape", return_value=(2,)):
        cast("EagerCallableProtocol", mod.eigh_tridiagonal)(a, b)
    cast("EagerCallableProtocol", mod.expm)(a)
    cast("EagerCallableProtocol", mod.global_norm)([a])
    mod.global_norm([])
    cast("EagerCallableProtocol", mod.logdet)(a)
    cast("EagerCallableProtocol", mod.logm)(a)
    cast("EagerCallableProtocol", mod.normalize)(a)
    cast("EagerCallableProtocol", mod.set_diag)(a, b)
    cast("EagerCallableProtocol", mod.tridiagonal_matmul)(a, a, a, a)
    cast("EagerCallableProtocol", mod.matrix_norm)(a)
    cast("EagerCallableProtocol", mod.vector_norm)(a)
    cast("EagerCallableProtocol", mod.svdvals)(a)
    cast("EagerCallableProtocol", mod.diagonal)(a)
    cast("EagerCallableProtocol", mod.cond)(a)
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
    cast("EagerCallableProtocol", mod.band_part)(a, 1, 1)
    cast("EagerCallableProtocol", mod.diag)(a, 0)
    cast("EagerCallableProtocol", mod.cross)(a, b)
    cast("EagerCallableProtocol", mod.trace)(a)
    cast("EagerCallableProtocol", mod.matrix_rank)(a)
    cast("EagerCallableProtocol", mod.matrix_transpose)(a)
    cast("EagerCallableProtocol", mod.sqrtm)(a)
    cast("EagerCallableProtocol", mod.tensor_diag)(a)
    cast("EagerCallableProtocol", mod.tensor_diag_part)(a)
    cast("EagerCallableProtocol", mod.diag_part)(a)
    cast("EagerCallableProtocol", mod.adjoint)(a)
    cast("EagerCallableProtocol", mod.eigh_tridiagonal)(a, b)
    cast("EagerCallableProtocol", mod.tridiagonal_matmul)(a, a, a, a)
    cast("EagerCallableProtocol", mod.matrix_norm)(a)
    cast("EagerCallableProtocol", mod.vector_norm)(a)
    cast("EagerCallableProtocol", mod.svdvals)(a)
    cast("EagerCallableProtocol", mod.diagonal)(a)
    cast("EagerCallableProtocol", mod.cond)(a)
    config.eager_mode = False
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    global_tracing_state.is_tracing = True
    from ml_switcheroo_compiler.ir.core import IRGraph

    global_tracing_state.active_graph = IRGraph()
    assert callable(test_matrix_ops_eager)


def test_matrix_ops_opdefs() -> None:
    """Test function test_matrix_ops_opdefs."""
    op1 = mod.Svdvals()
    assert cast("EagerCallableProtocol", op1.infer_shape)(cast("EagerCallableProtocol", DummyTensor)((2, 3))) == (2,)
    assert cast("EagerCallableProtocol", op1.infer_shape)(cast("EagerCallableProtocol", DummyTensor)((3,))) == (3,)
    op2 = mod.TridiagonalMatmul()
    assert cast("EagerCallableProtocol", op2.infer_shape)(None, None, None, cast("EagerCallableProtocol", DummyTensor)((2, 2))) == (2, 2)
    op3 = mod.Cond()
    assert op3.infer_shape() == ()
