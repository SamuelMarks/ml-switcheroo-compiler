# ruff: noqa: E501
import sys
from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.decompositions.svd import Svd, svd


def test_svd_dummy() -> None:
    op = Svd()
    assert op.infer_shape() == ()

    class MockTensor:
        shape = (2, 3, 4)

    t = MockTensor()
    (u_shape, s_shape, vh_shape) = op.infer_shape(t, compute_uv=True, full_matrices=True)
    assert u_shape == (2, 3, 3)
    assert s_shape == (2, 3)
    assert vh_shape == (2, 4, 4)
    (u_shape, s_shape, vh_shape) = op.infer_shape(t, compute_uv=True, full_matrices=False)
    assert u_shape == (2, 3, 3)
    assert s_shape == (2, 3)
    assert vh_shape == (2, 3, 4)
    res = op.infer_shape(t, compute_uv=False)
    assert res == ((2, 3),)


def test_svd_full_matrices_false() -> None:
    input_data = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "float32", "cpu"))
    config.eager_mode = False
    svd_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.svd"]
    with patch.object(svd_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = ("mock_U", "mock_S", "mock_Vh")
        res = svd(t, full_matrices=False)
        assert res == ("mock_U", "mock_S", "mock_Vh")
        mock_emit.assert_called_once()


def test_svd_eager() -> None:
    input_data = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "float32", "cpu"))
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, 1.0]), np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        mock_backend.return_value.array = lambda x: x
        (res1, res2, res3) = svd(t)
        assert isinstance(res1, Tensor)
        assert isinstance(res2, Tensor)
        assert isinstance(res3, Tensor)


def test_svd_full_matrices_true() -> None:
    input_data = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "float32", "cpu"))
    config.eager_mode = False
    svd_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.svd"]
    with patch.object(svd_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = ("mock_U", "mock_S", "mock_Vh")
        res = svd(t, full_matrices=True)
        assert res == ("mock_U", "mock_S", "mock_Vh")
        mock_emit.assert_called_once()
