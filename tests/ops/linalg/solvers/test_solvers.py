"""Tests for test_solvers."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.linalg.solvers as mod
from ml_switcheroo_compiler.core.config import config

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


class DummyTensor:
    """Test class DummyTensor."""

    def __init__(self, shape: EagerValue) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape
        self.data = typing.cast("EagerCallableProtocol", np.zeros)(shape)
        self.dtype = "float32"
        self.device = None


@patch("ml_switcheroo_compiler.ops.linalg.solvers._emit_linalg_node")
def test_solvers_tracing(mock_emit: EagerTensorProtocol) -> None:
    """Test function test_solvers_tracing.

    Args:
        mock_emit: Test parameter `mock_emit`.
    """
    config.eager_mode = False

    a = DummyTensor((2, 2))
    b = DummyTensor((2, 2))
    c = DummyTensor((2,))

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.CholeskySolve.infer_shape", return_value=(2, 2)):
        typing.cast("EagerCallableProtocol", mod.cholesky_solve)(a, b)

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.BandedTriangularSolve.infer_shape", return_value=(2, 2)):
        typing.cast("EagerCallableProtocol", mod.banded_triangular_solve)(a, b)

    typing.cast("EagerCallableProtocol", mod.conjugate_gradient)(a, b)

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.Lstsq.infer_shape", return_value=(2, 2)):
        typing.cast("EagerCallableProtocol", mod.lstsq)(a, b)

    typing.cast("EagerCallableProtocol", mod.lu)(a)

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.LuMatrixInverse.infer_shape", return_value=(2, 2)):
        typing.cast("EagerCallableProtocol", mod.lu_matrix_inverse)(a, b)

    with patch("ml_switcheroo_compiler.ops.linalg.solvers.LuReconstruct.infer_shape", return_value=(2, 2)):
        typing.cast("EagerCallableProtocol", mod.lu_reconstruct)(a, b)

    typing.cast("EagerCallableProtocol", mod.lu_solve)(a, b, c)

    with patch("ml_switcheroo_compiler.ops.linalg.decompositions.solvers.TriangularSolve.infer_shape", return_value=(2, 2)):
        typing.cast("EagerCallableProtocol", mod.triangular_solve)(a, b)

    typing.cast("EagerCallableProtocol", mod.tridiagonal_solve)(a, b)

    typing.cast("EagerCallableProtocol", mod.tensorinv)(a)
    typing.cast("EagerCallableProtocol", mod.tensorsolve)(a, b)
    assert c.shape == (2,)


@patch("ml_switcheroo_compiler.ops.linalg.solvers.get_active_backend")
def test_solvers_eager(mock_get_backend: EagerTensorProtocol) -> None:
    """Test function test_solvers_eager.

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
    c = DummyTensor((2,))

    typing.cast("EagerCallableProtocol", mod.cholesky_solve)(a, b)
    typing.cast("EagerCallableProtocol", mod.banded_triangular_solve)(a, b)

    typing.cast("EagerCallableProtocol", mod.lstsq)(a, b)
    mock_backend.execute_op.return_value = (np.zeros((2, 2)), np.zeros((2, 2)))
    typing.cast("EagerCallableProtocol", mod.lstsq)(a, b)
    mock_backend.execute_op.return_value = np.zeros((2, 2))

    typing.cast("EagerCallableProtocol", mod.lu_matrix_inverse)(a, b)
    typing.cast("EagerCallableProtocol", mod.lu_reconstruct)(a, b)
    typing.cast("EagerCallableProtocol", mod.triangular_solve)(a, b)
    typing.cast("EagerCallableProtocol", mod.tensorinv)(a)
    typing.cast("EagerCallableProtocol", mod.tensorsolve)(a, b)

    config.eager_mode = False
    assert c.shape == (2,)


def test_solvers_opdefs() -> None:
    # Lstsq
    """Test function test_solvers_opdefs."""
    op1 = mod.Lstsq()
    assert typing.cast("EagerCallableProtocol", op1.infer_shape)(typing.cast("EagerCallableProtocol", DummyTensor)((2, 2)), typing.cast("EagerCallableProtocol", DummyTensor)((2,))) == (2,)
    assert typing.cast("EagerCallableProtocol", op1.infer_shape)(typing.cast("EagerCallableProtocol", DummyTensor)((2, 2)), typing.cast("EagerCallableProtocol", DummyTensor)((2, 2))) == (2, 2)

    # Pinv
    op2 = mod.Pinv()
    assert typing.cast("EagerCallableProtocol", op2.infer_shape)(typing.cast("EagerCallableProtocol", DummyTensor)((2, 3))) == (3, 2)
    assert typing.cast("EagerCallableProtocol", op2.infer_shape)(typing.cast("EagerCallableProtocol", DummyTensor)((2,))) == (2,)
    assert typing.cast("EagerCallableProtocol", op2.infer_shape)("not_tensor") == ()

    # Sqrtm
    op3 = mod.Sqrtm()
    assert typing.cast("EagerCallableProtocol", op3.infer_shape)(typing.cast("EagerCallableProtocol", DummyTensor)((2, 2))) == (2, 2)

    # CholeskySolve
    op4 = mod.CholeskySolve()
    assert typing.cast("EagerCallableProtocol", op4.infer_shape)(typing.cast("EagerCallableProtocol", DummyTensor)((2, 2)), typing.cast("EagerCallableProtocol", DummyTensor)((2,))) == (2,)

    # BandedTriangularSolve
    op5 = mod.BandedTriangularSolve()
    assert typing.cast("EagerCallableProtocol", op5.infer_shape)(typing.cast("EagerCallableProtocol", DummyTensor)((2, 2)), typing.cast("EagerCallableProtocol", DummyTensor)((2,))) == (2,)

    # EighTridiagonal
    op6 = mod.EighTridiagonal()
    assert typing.cast("EagerCallableProtocol", op6.infer_shape)(typing.cast("EagerCallableProtocol", DummyTensor)((2,)), typing.cast("EagerCallableProtocol", DummyTensor)((1,))) == ((2,), [2, 2])

    # MatrixNorm
    op7 = mod.MatrixNorm()
    assert typing.cast("EagerCallableProtocol", op7.infer_shape)(typing.cast("EagerCallableProtocol", DummyTensor)((2, 2))) == ()

    # VectorNorm
    op8 = mod.VectorNorm()
    assert typing.cast("EagerCallableProtocol", op8.infer_shape)(typing.cast("EagerCallableProtocol", DummyTensor)((2,))) == ()

    # Svdvals
    op9 = mod.Svdvals()
    assert typing.cast("EagerCallableProtocol", op9.infer_shape)(typing.cast("EagerCallableProtocol", DummyTensor)((2, 2))) == (2,)
    assert typing.cast("EagerCallableProtocol", op9.infer_shape)(typing.cast("EagerCallableProtocol", DummyTensor)((2,))) == ()

    # Tensorinv
    op10 = mod.Tensorinv()
    assert typing.cast("EagerCallableProtocol", op10.infer_shape)(typing.cast("EagerCallableProtocol", DummyTensor)((2, 2))) == (2, 2)

    # Tensorsolve
    op11 = mod.Tensorsolve()
    assert typing.cast("EagerCallableProtocol", op11.infer_shape)(typing.cast("EagerCallableProtocol", DummyTensor)((2, 2, 2)), typing.cast("EagerCallableProtocol", DummyTensor)((2,))) == (2, 2)

    # LuMatrixInverse
    op12 = mod.LuMatrixInverse()
    assert typing.cast("EagerCallableProtocol", op12.infer_shape)(typing.cast("EagerCallableProtocol", DummyTensor)((2, 2)), typing.cast("EagerCallableProtocol", DummyTensor)((2,))) == (2, 2)

    # LuReconstruct
    op13 = mod.LuReconstruct()
    assert typing.cast("EagerCallableProtocol", op13.infer_shape)(typing.cast("EagerCallableProtocol", DummyTensor)((2, 2)), typing.cast("EagerCallableProtocol", DummyTensor)((2,))) == (2, 2)
