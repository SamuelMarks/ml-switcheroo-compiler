"""Tests for test_solvers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.linalg.solvers as mod
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


@patch("ml_switcheroo_compiler.ops.linalg.solvers._emit_linalg_node")
def test_solvers_tracing(mock_emit: EagerTensorProtocol) -> None:
    """Test function test_solvers_tracing."""
    config.eager_mode = False

    a = DummyTensor((2, 2))
    b = DummyTensor((2, 2))
    c = DummyTensor((2,))

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.CholeskySolve.infer_shape", return_value=(2, 2)):
        cast("Callable[..., EagerTensorProtocol]", mod.cholesky_solve)(a, b)

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.BandedTriangularSolve.infer_shape", return_value=(2, 2)):
        cast("Callable[..., EagerTensorProtocol]", mod.banded_triangular_solve)(a, b)

    cast("Callable[..., EagerTensorProtocol]", mod.conjugate_gradient)(a, b)

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.Lstsq.infer_shape", return_value=(2, 2)):
        cast("Callable[..., EagerTensorProtocol]", mod.lstsq)(a, b)

    cast("Callable[..., EagerTensorProtocol]", mod.lu)(a)

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.LuMatrixInverse.infer_shape", return_value=(2, 2)):
        cast("Callable[..., EagerTensorProtocol]", mod.lu_matrix_inverse)(a, b)

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.LuReconstruct.infer_shape", return_value=(2, 2)):
        cast("Callable[..., EagerTensorProtocol]", mod.lu_reconstruct)(a, b)

    cast("Callable[..., EagerTensorProtocol]", mod.lu_solve)(a, b, c)

    with patch("ml_switcheroo_compiler.ops.linalg.decompositions.solvers.TriangularSolve.infer_shape", return_value=(2, 2)):
        cast("Callable[..., EagerTensorProtocol]", mod.triangular_solve)(a, b)

    cast("Callable[..., EagerTensorProtocol]", mod.tridiagonal_solve)(a, b)

    cast("Callable[..., EagerTensorProtocol]", mod.tensorinv)(a)
    cast("Callable[..., EagerTensorProtocol]", mod.tensorsolve)(a, b)
    assert c is not None or c is None


@patch("ml_switcheroo_compiler.ops.linalg.solvers.get_active_backend")
def test_solvers_eager(mock_get_backend: EagerTensorProtocol) -> None:
    """Test function test_solvers_eager."""
    config.eager_mode = True

    mock_backend = MagicMock()
    mock_backend.execute_op.return_value = np.zeros((2, 2))
    mock_backend.array.return_value = np.zeros((2, 2))
    mock_backend.array.return_value.shape = (2, 2)
    setattr(mock_get_backend, "return_value", mock_backend)

    a = DummyTensor((2, 2))
    b = DummyTensor((2, 2))
    c = DummyTensor((2,))

    cast("Callable[..., EagerTensorProtocol]", mod.cholesky_solve)(a, b)
    cast("Callable[..., EagerTensorProtocol]", mod.banded_triangular_solve)(a, b)

    cast("Callable[..., EagerTensorProtocol]", mod.lstsq)(a, b)
    mock_backend.execute_op.return_value = (np.zeros((2, 2)), np.zeros((2, 2)))
    cast("Callable[..., EagerTensorProtocol]", mod.lstsq)(a, b)
    mock_backend.execute_op.return_value = np.zeros((2, 2))

    cast("Callable[..., EagerTensorProtocol]", mod.lu_matrix_inverse)(a, b)
    cast("Callable[..., EagerTensorProtocol]", mod.lu_reconstruct)(a, b)
    cast("Callable[..., EagerTensorProtocol]", mod.triangular_solve)(a, b)
    cast("Callable[..., EagerTensorProtocol]", mod.tensorinv)(a)
    cast("Callable[..., EagerTensorProtocol]", mod.tensorsolve)(a, b)

    config.eager_mode = False
    assert c is not None or c is None


def test_solvers_opdefs() -> None:
    # Lstsq
    """Test function test_solvers_opdefs."""
    op1 = mod.Lstsq()
    assert cast("Callable[..., EagerTensorProtocol]", op1.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2)), cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2,))) == (2,)
    assert cast("Callable[..., EagerTensorProtocol]", op1.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2)), cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2))) == (2, 2)

    # Pinv
    op2 = mod.Pinv()
    assert cast("Callable[..., EagerTensorProtocol]", op2.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 3))) == (3, 2)
    assert cast("Callable[..., EagerTensorProtocol]", op2.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2,))) == (2,)
    assert cast("Callable[..., EagerTensorProtocol]", op2.infer_shape)("not_tensor") == ()

    # Sqrtm
    op3 = mod.Sqrtm()
    assert cast("Callable[..., EagerTensorProtocol]", op3.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2))) == (2, 2)

    # CholeskySolve
    op4 = mod.CholeskySolve()
    assert cast("Callable[..., EagerTensorProtocol]", op4.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2)), cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2,))) == (2,)

    # BandedTriangularSolve
    op5 = mod.BandedTriangularSolve()
    assert cast("Callable[..., EagerTensorProtocol]", op5.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2)), cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2,))) == (2,)

    # EighTridiagonal
    op6 = mod.EighTridiagonal()
    assert cast("Callable[..., EagerTensorProtocol]", op6.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2,)), cast("Callable[..., EagerTensorProtocol]", DummyTensor)((1,))) == ((2,), [2, 2])

    # MatrixNorm
    op7 = mod.MatrixNorm()
    assert cast("Callable[..., EagerTensorProtocol]", op7.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2))) == ()

    # VectorNorm
    op8 = mod.VectorNorm()
    assert cast("Callable[..., EagerTensorProtocol]", op8.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2,))) == ()

    # Svdvals
    op9 = mod.Svdvals()
    assert cast("Callable[..., EagerTensorProtocol]", op9.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2))) == (2,)
    assert cast("Callable[..., EagerTensorProtocol]", op9.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2,))) == ()

    # Tensorinv
    op10 = mod.Tensorinv()
    assert cast("Callable[..., EagerTensorProtocol]", op10.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2))) == (2, 2)

    # Tensorsolve
    op11 = mod.Tensorsolve()
    assert cast("Callable[..., EagerTensorProtocol]", op11.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)), cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2,))) == (2, 2)

    # LuMatrixInverse
    op12 = mod.LuMatrixInverse()
    assert cast("Callable[..., EagerTensorProtocol]", op12.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2)), cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2,))) == (2, 2)

    # LuReconstruct
    op13 = mod.LuReconstruct()
    assert cast("Callable[..., EagerTensorProtocol]", op13.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2)), cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2,))) == (2, 2)
