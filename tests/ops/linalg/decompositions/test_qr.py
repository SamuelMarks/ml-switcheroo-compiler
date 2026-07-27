# ruff: noqa: E501
import sys
from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.decompositions.qr import Hessenberg, HouseholderProduct, Qdwh, Qr, Schur, Tridiagonal, hessenberg, householder_product, qdwh, qr, schur, tridiagonal


def test_qr_dummy() -> None:
    assert Qr().infer_shape() == ()

    class MockTensor:
        shape = (2, 4, 3)

    t = MockTensor()
    assert Qr().infer_shape(t, mode="complete") == ((2, 4, 4), (2, 4, 3))
    assert Qr().infer_shape(t, mode="r") == ((2, 3, 3),)
    assert Qr().infer_shape(t, mode="reduced") == ((2, 4, 3), (2, 3, 3))
    assert Hessenberg().infer_shape(t) == ((2, 4, 3), (2, 4, 3))
    assert HouseholderProduct().infer_shape(t) == ()
    assert Schur().infer_shape(t) == ((2, 4, 3), (2, 4, 3))
    assert Tridiagonal().infer_shape(t) == ((2, 4), (2, 2), (2, 4, 3))
    assert Qdwh().infer_shape(t) == ((2, 4, 3), (2, 4, 3), (2,), (2,))
    t2 = MockTensor()
    t2.shape = (2, 0)
    assert Tridiagonal().infer_shape(t2) == ((2,), (2,), (2, 0))


def test_qr_ops() -> None:
    input_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "float32", "cpu"))
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = (np.zeros((2, 2)), np.zeros((2, 2)))
        (res1, res2) = qr(t)
        assert isinstance(res1, Tensor)
        mock_backend.return_value.execute_op.return_value = (np.zeros((2, 2)), np.zeros((2, 2)))
        (res1, res2) = hessenberg(t)
        assert isinstance(res1, Tensor)
        mock_backend.return_value.execute_op.return_value = np.zeros((2, 2))
        res = householder_product(t, t)
        assert isinstance(res, Tensor)
        mock_backend.return_value.execute_op.return_value = (np.zeros((2, 2)), np.zeros((2, 2)))
        (res1, res2) = schur(t)
        assert isinstance(res1, Tensor)
        mock_backend.return_value.execute_op.return_value = (np.zeros((2,)), np.zeros((1,)), np.zeros((2, 2)))
        (res1, res2, res3) = tridiagonal(t)
        assert isinstance(res1, Tensor)
        mock_backend.return_value.execute_op.return_value = (np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(()), np.zeros(()))
        (res1, res2, res3, res4) = qdwh(t)
        assert isinstance(res1, Tensor)
    config.eager_mode = False
    qr_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.qr"]
    with patch.object(qr_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = ("mock_q", "mock_r")
        qr(t)
        qr(t, mode="complete")
        mock_emit.return_value = ("mock_h", "mock_q")
        hessenberg(t)
        mock_emit.return_value = "mock_h"
        householder_product(t, t)
        mock_emit.return_value = ("mock_t", "mock_z")
        schur(t)
        mock_emit.return_value = ("mock_diag", "mock_off", "mock_q")
        tridiagonal(t)
        t_zero = Tensor(np.zeros((0, 0)), TensorConfig((0, 0), "float32", "cpu"))
        tridiagonal(t_zero)
        mock_emit.return_value = ("mock_q", "mock_h", "mock_iters", "mock_conv")
        qdwh(t)
