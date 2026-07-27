# ruff: noqa: E501
import sys
from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.decompositions.inv import InvEx, inv_ex, pinv


def test_inv_ex_dummy() -> None:
    assert InvEx().infer_shape() == ()


def test_inv_ex() -> None:
    input_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "float32", "cpu"))
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = (np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), np.array(0, dtype=np.int32))
        mock_backend.return_value.array = lambda x: x
        (res_inv, res_info) = inv_ex(t, check_errors=True)
        assert isinstance(res_inv, Tensor)
        assert isinstance(res_info, Tensor)
    config.eager_mode = False
    inv_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.inv"]
    with patch.object(inv_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = ("mock_inv", "mock_info")
        (res_inv, res_info) = inv_ex(t, check_errors=False)
        assert res_inv == "mock_inv"
        mock_emit.assert_called_once()


def test_pinv() -> None:
    input_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "float32", "cpu"))
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        mock_backend.return_value.array = lambda x: x
        res = pinv(t, rcond=1e-15)
        assert isinstance(res, Tensor)
    config.eager_mode = False
    inv_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.inv"]
    with patch.object(inv_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = "mock_pinv"
        res = pinv(t, rcond=1e-15)
        assert res == "mock_pinv"
        mock_emit.assert_called_once()


def test_inv_and_tri_inv() -> None:
    from ml_switcheroo_compiler.ops.linalg.decompositions.inv import Inv, TriInv, inv, tri_inv

    assert Inv().infer_shape() == ()
    assert TriInv().infer_shape() == ()
    input_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "float32", "cpu"))
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        mock_backend.return_value.array = lambda x: x
        res1 = inv(t)
        assert isinstance(res1, Tensor)
        mock_backend.return_value.execute_op.return_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        res2 = tri_inv(t, lower=True)
        assert isinstance(res2, Tensor)
    config.eager_mode = False
    inv_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.inv"]
    with patch.object(inv_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = "mock_inv"
        res1 = inv(t)
        assert res1 == "mock_inv"
        mock_emit.return_value = "mock_tri_inv"
        res2 = tri_inv(t, lower=True)
        assert res2 == "mock_tri_inv"
