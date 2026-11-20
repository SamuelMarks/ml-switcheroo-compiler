"""Tests for dense and triangular linear solver dispatch (`solve`, `solve_ex`, `solve_triangular`)."""

from __future__ import annotations

import typing
from unittest.mock import patch

import numpy as np

import ml_switcheroo_compiler.ops.linalg.solvers as solve_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.solvers import Solve, SolveEx, solve, solve_ex, solve_triangular
from tests.testing_utils import unpack_pair

if typing.TYPE_CHECKING:
    pass


def test_solve_ex_dummy() -> None:
    """Verify `SolveEx.infer_shape()` returns `()` when no operands are provided."""
    assert SolveEx().infer_shape() == ()


def test_solve_ex() -> None:
    """Verify eager execution and symbolic node emission for `solve_ex`."""
    a_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b_data = np.array([1.0, 2.0], dtype=np.float32)
    a = Tensor(a_data, TensorConfig(a_data.shape, "float32", "cpu"))
    b = Tensor(b_data, TensorConfig(b_data.shape, "float32", "cpu"))
    config.eager_mode = True
    with patch.object(solve_mod, "get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = (
            np.array([1.0, 2.0], dtype=np.float32),
            np.array(0, dtype=np.int32),
        )
        (res_sol, res_info) = unpack_pair(solve_ex(a, b, check_errors=True))
        assert isinstance(res_sol, Tensor)
        assert isinstance(res_info, Tensor)
    config.eager_mode = False
    with patch.object(solve_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = ("mock_sol", "mock_info")
        (res_sol, res_info) = unpack_pair(solve_ex(a, b, check_errors=False))
        assert res_sol == "mock_sol"
        mock_emit.assert_called_once()


def test_solve_dummy() -> None:
    """Verify `Solve.infer_shape()` returns `()` when no operands are provided."""
    assert Solve().infer_shape() == ()


def test_solve() -> None:
    """Verify eager execution and symbolic node emission for `solve`."""
    a_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b_data = np.array([1.0, 2.0], dtype=np.float32)
    a = Tensor(a_data, TensorConfig(a_data.shape, "float32", "cpu"))
    b = Tensor(b_data, TensorConfig(b_data.shape, "float32", "cpu"))
    config.eager_mode = True
    with patch.object(solve_mod, "get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.array([1.0, 2.0], dtype=np.float32)
        res = solve(a, b)
        assert isinstance(res, Tensor)
    config.eager_mode = False
    with patch.object(solve_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = "mock_sol"
        res = solve(a, b)
        assert res == "mock_sol"


def test_solve_triangular() -> None:
    """Verify eager execution and symbolic node emission for `solve_triangular`."""
    a_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b_data = np.array([1.0, 2.0], dtype=np.float32)
    a = Tensor(a_data, TensorConfig(a_data.shape, "float32", "cpu"))
    b = Tensor(b_data, TensorConfig(b_data.shape, "float32", "cpu"))
    config.eager_mode = True
    with patch.object(solve_mod, "get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.array([1.0, 2.0], dtype=np.float32)
        res = solve_triangular(a, b)
        assert isinstance(res, Tensor)
    config.eager_mode = False
    with patch.object(solve_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = "mock_sol"
        res = solve_triangular(a, b)
        assert res == "mock_sol"
