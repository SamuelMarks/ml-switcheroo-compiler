# ruff: noqa: E501
import sys
from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.decompositions.solve import SolveEx, solve_ex


def test_solve_ex_dummy() -> None:
    assert SolveEx().infer_shape() == ()


def test_solve_ex() -> None:
    a_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b_data = np.array([1.0, 2.0], dtype=np.float32)
    a = Tensor(a_data, TensorConfig(a_data.shape, "float32", "cpu"))
    b = Tensor(b_data, TensorConfig(b_data.shape, "float32", "cpu"))
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = (np.array([1.0, 2.0], dtype=np.float32), np.array(0, dtype=np.int32))
        mock_backend.return_value.array = lambda x: x
        (res_sol, res_info) = solve_ex(a, b, check_errors=True)
        assert isinstance(res_sol, Tensor)
        assert isinstance(res_info, Tensor)
    config.eager_mode = False
    solve_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.solve"]
    with patch.object(solve_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = ("mock_sol", "mock_info")
        (res_sol, res_info) = solve_ex(a, b, check_errors=False)
        assert res_sol == "mock_sol"
        mock_emit.assert_called_once()


def test_solve_dummy() -> None:
    from ml_switcheroo_compiler.ops.linalg.decompositions.solve import Solve

    assert Solve().infer_shape() == ()


def test_solve() -> None:
    from ml_switcheroo_compiler.ops.linalg.decompositions.solve import solve

    a_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b_data = np.array([1.0, 2.0], dtype=np.float32)
    a = Tensor(a_data, TensorConfig(a_data.shape, "float32", "cpu"))
    b = Tensor(b_data, TensorConfig(b_data.shape, "float32", "cpu"))
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.array([1.0, 2.0], dtype=np.float32)
        res = solve(a, b)
        assert isinstance(res, Tensor)
    config.eager_mode = False
    solve_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.solve"]
    with patch.object(solve_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = "mock_sol"
        res = solve(a, b)
        assert res == "mock_sol"


def test_solve_triangular() -> None:
    from ml_switcheroo_compiler.ops.linalg.decompositions.solve import solve_triangular

    a_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b_data = np.array([1.0, 2.0], dtype=np.float32)
    a = Tensor(a_data, TensorConfig(a_data.shape, "float32", "cpu"))
    b = Tensor(b_data, TensorConfig(b_data.shape, "float32", "cpu"))
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.array([1.0, 2.0], dtype=np.float32)
        res = solve_triangular(a, b)
        assert isinstance(res, Tensor)
    config.eager_mode = False
    solve_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.solve"]
    with patch.object(solve_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = "mock_sol"
        res = solve_triangular(a, b)
        assert res == "mock_sol"
