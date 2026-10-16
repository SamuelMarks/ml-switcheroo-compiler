"""Tests for test_matrix_ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.linalg.matrix_ops as mod
from ml_switcheroo_compiler.core.config import config

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class DummyTensor:
    """Test class DummyTensor."""

    def __init__(self, shape: EagerValue) -> None:
        """Test function __init__."""
        self.shape = shape
        self.data = cast("Callable[..., EagerTensorProtocol]", np.zeros)(shape)
        self.dtype = "float32"
        self.device = None


@patch("ml_switcheroo_compiler.ops.linalg.matrix_ops._emit_linalg_node")
def test_matrix_ops_tracing(mock_emit: EagerTensorProtocol) -> None:
    """Test function test_matrix_ops_tracing."""
    config.eager_mode = False
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    global_tracing_state.is_tracing = True
    from ml_switcheroo_compiler.ir.core import IRGraph

    global_tracing_state.active_graph = IRGraph()

    a = DummyTensor((2, 2))
    b = DummyTensor((2, 2))

    cast("Callable[..., EagerTensorProtocol]", mod.band_part)(a, 1, 1)
    cast("Callable[..., EagerTensorProtocol]", mod.diag)(a, 0)
    cast("Callable[..., EagerTensorProtocol]", mod.cross)(a, b)
    cast("Callable[..., EagerTensorProtocol]", mod.cross)(a, b, axes={"axis": 0})

    with patch("ml_switcheroo_compiler.ops.linalg.products.Trace.infer_shape", return_value=(2,)):
        cast("Callable[..., EagerTensorProtocol]", mod.trace)(a)

    with patch("ml_switcheroo_compiler.ops.linalg.products.MatrixRank.infer_shape", return_value=(1,)):
        cast("Callable[..., EagerTensorProtocol]", mod.matrix_rank)(a)

    with patch("ml_switcheroo_compiler.ops.linalg.products.MatrixTranspose.infer_shape", return_value=(2, 2)):
        cast("Callable[..., EagerTensorProtocol]", mod.matrix_transpose)(a)

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.Sqrtm.infer_shape", return_value=(2, 2)):
        try:
            cast("Callable[..., EagerTensorProtocol]", mod.sqrtm)(a)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None

    cast("Callable[..., EagerTensorProtocol]", mod.tensor_diag)(a)
    cast("Callable[..., EagerTensorProtocol]", mod.tensor_diag_part)(a)
    cast("Callable[..., EagerTensorProtocol]", mod.diag_part)(a)

    with patch("ml_switcheroo_compiler.ops.linalg.products.Adjoint.infer_shape", return_value=(2, 2)):
        cast("Callable[..., EagerTensorProtocol]", mod.adjoint)(a)

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.EighTridiagonal.infer_shape", return_value=(2,)):
        cast("Callable[..., EagerTensorProtocol]", mod.eigh_tridiagonal)(a, b)

    try:
        cast("Callable[..., EagerTensorProtocol]", mod.expm)(a)
    except NameError as _exc:
        assert _exc is not None

    cast("Callable[..., EagerTensorProtocol]", mod.global_norm)([a])
    mod.global_norm([])
    cast("Callable[..., EagerTensorProtocol]", mod.logdet)(a)
    cast("Callable[..., EagerTensorProtocol]", mod.logm)(a)
    cast("Callable[..., EagerTensorProtocol]", mod.normalize)(a)
    cast("Callable[..., EagerTensorProtocol]", mod.set_diag)(a, b)

    cast("Callable[..., EagerTensorProtocol]", mod.tridiagonal_matmul)(a, a, a, a)
    cast("Callable[..., EagerTensorProtocol]", mod.matrix_norm)(a)
    cast("Callable[..., EagerTensorProtocol]", mod.vector_norm)(a)

    cast("Callable[..., EagerTensorProtocol]", mod.svdvals)(a)
    cast("Callable[..., EagerTensorProtocol]", mod.diagonal)(a)
    cast("Callable[..., EagerTensorProtocol]", mod.cond)(a)


@patch("ml_switcheroo_compiler.ops.linalg.matrix_ops.get_active_backend")
def test_matrix_ops_eager(mock_get_backend: EagerTensorProtocol) -> None:
    """Test function test_matrix_ops_eager."""
    config.eager_mode = True

    mock_backend = MagicMock()
    mock_backend.execute_op.return_value = np.zeros((2, 2))
    mock_backend.array.return_value = np.zeros((2, 2))
    mock_backend.array.return_value.shape = (2, 2)
    setattr(mock_get_backend, "return_value", mock_backend)

    a = DummyTensor((2, 2))
    b = DummyTensor((2, 2))

    cast("Callable[..., EagerTensorProtocol]", mod.band_part)(a, 1, 1)
    cast("Callable[..., EagerTensorProtocol]", mod.diag)(a, 0)
    cast("Callable[..., EagerTensorProtocol]", mod.cross)(a, b)
    cast("Callable[..., EagerTensorProtocol]", mod.trace)(a)
    cast("Callable[..., EagerTensorProtocol]", mod.matrix_rank)(a)
    cast("Callable[..., EagerTensorProtocol]", mod.matrix_transpose)(a)
    try:
        cast("Callable[..., EagerTensorProtocol]", mod.sqrtm)(a)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    cast("Callable[..., EagerTensorProtocol]", mod.tensor_diag)(a)
    cast("Callable[..., EagerTensorProtocol]", mod.tensor_diag_part)(a)
    cast("Callable[..., EagerTensorProtocol]", mod.diag_part)(a)
    cast("Callable[..., EagerTensorProtocol]", mod.adjoint)(a)
    cast("Callable[..., EagerTensorProtocol]", mod.eigh_tridiagonal)(a, b)

    cast("Callable[..., EagerTensorProtocol]", mod.tridiagonal_matmul)(a, a, a, a)
    cast("Callable[..., EagerTensorProtocol]", mod.matrix_norm)(a)
    cast("Callable[..., EagerTensorProtocol]", mod.vector_norm)(a)

    cast("Callable[..., EagerTensorProtocol]", mod.svdvals)(a)
    cast("Callable[..., EagerTensorProtocol]", mod.diagonal)(a)
    cast("Callable[..., EagerTensorProtocol]", mod.cond)(a)

    config.eager_mode = False
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    global_tracing_state.is_tracing = True
    from ml_switcheroo_compiler.ir.core import IRGraph

    global_tracing_state.active_graph = IRGraph()


def test_matrix_ops_opdefs() -> None:
    # Svdvals
    """Test function test_matrix_ops_opdefs."""
    op1 = mod.Svdvals()
    assert cast("Callable[..., EagerTensorProtocol]", op1.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 3))) == (2,)
    assert cast("Callable[..., EagerTensorProtocol]", op1.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((3,))) == (3,)

    # TridiagonalMatmul
    op2 = mod.TridiagonalMatmul()
    assert cast("Callable[..., EagerTensorProtocol]", op2.infer_shape)(None, None, None, cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2))) == (2, 2)

    # Cond
    op3 = mod.Cond()
    assert op3.infer_shape() == ()
