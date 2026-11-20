"""Tests for consolidated linear-system solvers and moved decomposition OpDefs."""

from __future__ import annotations

import sys
import typing
from unittest.mock import patch

import numpy as np

import ml_switcheroo_compiler.ops.linalg.solvers as solvers_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.decompositions.lu import Lu, lu
from ml_switcheroo_compiler.ops.linalg.decompositions.norms import Cross, MatrixExponential, Norm, PowerIteration
from ml_switcheroo_compiler.ops.linalg.decompositions.svd import Polar, polar
from ml_switcheroo_compiler.ops.linalg.products import TridiagonalMatmul
from ml_switcheroo_compiler.ops.linalg.solvers import (
    LuSolve,
    TriangularSolve,
    TridiagonalSolve,
    lu_solve,
    tridiagonal_solve,
)
from tests.testing_utils import MockTestStub, unpack_pair

lu_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.lu"]
svd_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.svd"]

if typing.TYPE_CHECKING:
    pass


def test_misc_infer_shapes() -> None:
    """Verify shape inference for `PowerIteration`, `Polar`, `TridiagonalSolve`, and `TridiagonalMatmul`."""

    class MockTensor(MockTestStub):
        """Mock tensor shape/dtype container."""

        shape = (2, 3, 4)
        dtype = "float32"

    t = MockTensor()
    (shapes, dtypes) = unpack_pair(PowerIteration().infer_shape(t))
    assert shapes == ((2, 4), (2, 3), (2,))
    assert dtypes == ("float32", "float32", "float32")
    assert Polar().infer_shape(t) == ((2, 3, 4), (2, 3, 4))
    assert TridiagonalSolve().infer_shape(t, t, t, t) == (2, 3, 4)
    assert TridiagonalMatmul().infer_shape(t, t, t, t) == (2, 3, 4)


def test_misc_eager_and_trace() -> None:
    """Verify eager execution and symbolic node emission for `lu`, `lu_solve`, `polar`, and `tridiagonal_solve`."""
    t = Tensor(np.zeros((2, 2)), TensorConfig((2, 2), "float32", "cpu"))
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        from tests.testing_utils import coerce_operand_list

        mock_backend.return_value.execute_op.return_value = (np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)))
        (res1, res2, res3) = coerce_operand_list(lu(t))
        assert isinstance(res1, Tensor)
        assert isinstance(res2, Tensor)
        assert isinstance(res3, Tensor)
    with patch.object(solvers_mod, "get_active_backend") as mock_solv_be:
        mock_solv_be.return_value.execute_op.return_value = np.zeros((2, 2))
        res = lu_solve((t, t), t)
        assert isinstance(res, Tensor)
        res = tridiagonal_solve(t, t, t, t)
        assert isinstance(res, Tensor)
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = (np.zeros((2, 2)), np.zeros((2, 2)))
        (res1, res2) = unpack_pair(polar(t))
        assert isinstance(res1, Tensor)
        assert isinstance(res2, Tensor)

    config.eager_mode = False
    with patch.object(lu_mod, "_emit_linalg_node", return_value=("mock_l", "mock_u", "mock_p")) as mock_lu_emit:
        assert lu(t) == ("mock_l", "mock_u", "mock_p")
        mock_lu_emit.assert_called_once()
    with patch.object(svd_mod, "_emit_linalg_node", return_value=("mock_u", "mock_p")) as mock_svd_emit:
        assert polar(t) == ("mock_u", "mock_p")
        mock_svd_emit.assert_called_once()
    with patch.object(solvers_mod, "_emit_linalg_node", return_value="mock_sol") as mock_solv_emit:
        assert lu_solve((t, t), t) == "mock_sol"
        assert tridiagonal_solve(t, t, t, t) == "mock_sol"
        assert mock_solv_emit.call_count == 2


def test_misc_dummy_infer_shapes() -> None:
    """Verify empty-argument fallback shape inference across decomposition and solver OpDefs."""
    assert TriangularSolve().infer_shape() == ()
    assert Lu().infer_shape() == ()
    assert LuSolve().infer_shape() == ()
    assert Norm().infer_shape() == ()
    assert MatrixExponential().infer_shape() == ()
    assert Cross().infer_shape() == ()
