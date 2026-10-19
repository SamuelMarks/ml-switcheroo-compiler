"""Tests for test_solvers."""

from __future__ import annotations

# ruff: noqa: E501
import sys
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.decompositions.solvers import Polar, PowerIteration, TridiagonalMatmul, TridiagonalSolve, lu, lu_solve, polar, tridiagonal_solve

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_misc_infer_shapes() -> None:
    """Test function test_misc_infer_shapes."""

    class MockTensor:
        """Test class MockTensor."""

        shape = (2, 3, 4)
        dtype = "float32"

    t = MockTensor()
    (shapes, dtypes) = cast("EagerCallableProtocol", PowerIteration)().infer_shape(t)
    assert shapes == ((2, 4), (2, 3), (2,))
    assert cast("EagerCallableProtocol", Polar)().infer_shape(t) == ((2, 3, 4), (2, 3, 4))
    assert cast("EagerCallableProtocol", TridiagonalSolve)().infer_shape(t, t, t, t) == (2, 3, 4)
    assert cast("EagerCallableProtocol", TridiagonalMatmul)().infer_shape(t, t, t, t) == (2, 3, 4)


def test_misc_eager_and_trace() -> None:
    """Test function test_misc_eager_and_trace."""
    t = Tensor(np.zeros((2, 2)), TensorConfig((2, 2), "float32", "cpu"))
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = (np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)))
        (res1, res2, res3) = cast("EagerCallableProtocol", lu)(t)
        assert isinstance(res1, Tensor)
        mock_backend.return_value.execute_op.return_value = np.zeros((2, 2))
        res = lu_solve((t, t), t)
        assert isinstance(res, Tensor)
        mock_backend.return_value.execute_op.return_value = (np.zeros((2, 2)), np.zeros((2, 2)))
        (res1, res2) = cast("EagerCallableProtocol", polar)(t)
        assert isinstance(res1, Tensor)
        mock_backend.return_value.execute_op.return_value = np.zeros((2, 2))
        res = tridiagonal_solve(t, t, t, t)
        assert isinstance(res, Tensor)
    config.eager_mode = False
    misc_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.solvers"]
    with patch.object(misc_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = ("mock_l", "mock_u", "mock_p")
        lu(t)
        mock_emit.return_value = "mock_sol"
        lu_solve((t, t), t)
        mock_emit.return_value = ("mock_u", "mock_p")
        polar(t)
        mock_emit.return_value = "mock_sol"
        tridiagonal_solve(t, t, t, t)


def test_misc_dummy_infer_shapes() -> None:
    """Test function test_misc_dummy_infer_shapes."""
    from ml_switcheroo_compiler.ops.linalg.decompositions.solvers import Cross, Lu, LuSolve, MatrixExponential, Norm, TriangularSolve

    assert TriangularSolve().infer_shape() == ()
    assert Lu().infer_shape() == ()
    assert LuSolve().infer_shape() == ()
    assert Norm().infer_shape() == ()
    assert MatrixExponential().infer_shape() == ()
    assert Cross().infer_shape() == ()
