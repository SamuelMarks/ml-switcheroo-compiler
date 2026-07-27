# ruff: noqa: E501
import sys
from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.decompositions.norms import _norm_out_shape, matrix_exp, matrix_exponential, matrix_power, norm, power_iteration


def test_norm_out_shape() -> None:
    assert _norm_out_shape((2, 3), None, keepdims=True) == (1, 1)
    assert _norm_out_shape((2, 3), None, keepdims=False) == ()
    assert _norm_out_shape((2, 3), 0, keepdims=False) == (3,)


def test_norms_eager() -> None:
    input_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "float32", "cpu"))
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.zeros((2, 2))
        res = matrix_power(t, 2)
        assert isinstance(res, Tensor)
        res = norm(t, ord=2, axis=None, keepdims=False)
        assert isinstance(res, Tensor)
        res = matrix_exponential(t)
        assert isinstance(res, Tensor)
        res = matrix_exp(t)
        assert isinstance(res, Tensor)
        mock_backend.return_value.execute_op.return_value = (np.zeros((2,)), np.zeros((2,)), np.zeros(()))
        (res1, res2, res3) = power_iteration(t, num_iters=1, u=None)
        assert isinstance(res1, Tensor)
        (res1, res2, res3) = power_iteration(t, num_iters=1, u=t)
        assert isinstance(res1, Tensor)


def test_norms_trace() -> None:
    input_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "float32", "cpu"))
    config.eager_mode = False
    norms_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.norms"]
    with patch.object(norms_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = "mock"
        assert matrix_power(t, 2) == "mock"
        assert norm(t, ord=2) == "mock"
        assert matrix_exponential(t) == "mock"
        assert matrix_exp(t) == "mock"
        mock_emit.return_value = ("mock1", "mock2", "mock3")
        assert power_iteration(t, num_iters=1, u=None) == ("mock1", "mock2", "mock3")
        assert power_iteration(t, num_iters=1, u=t) == ("mock1", "mock2", "mock3")
