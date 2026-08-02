# ruff: noqa: E501
import sys

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig

gru_mod = sys.modules["ml_switcheroo_compiler.ops.nn.gru"]
import ml_switcheroo_compiler.ops.nn.linear_ops as linear


def test_linear_ops_coverage():
    config.eager_mode = True
    t_in = Tensor(np.ones((1, 2)), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    w = Tensor(np.ones((3, 2)), TensorConfig(shape=(3, 2), dtype=DType("float32"), device=Device("cpu")))
    b = Tensor(np.ones((3,)), TensorConfig(shape=(3,), dtype=DType("float32"), device=Device("cpu")))

    def mock_matmul(*args):
        return "mm"

    def mock_add(*args):
        return "add"

    orig_matmul = linear.matmul
    orig_add = linear.add
    linear.matmul = mock_matmul
    linear.add = mock_add
    try:
        assert linear.linear(t_in, w) == "mm"
        assert linear.linear(t_in, w, b) == "add"
    finally:
        linear.matmul = orig_matmul
        linear.add = orig_add

    t_in1 = Tensor(np.ones((1, 2)), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    t_in2 = Tensor(np.ones((1, 4)), TensorConfig(shape=(1, 4), dtype=DType("float32"), device=Device("cpu")))
    w_bi = Tensor(np.ones((3, 2, 4)), TensorConfig(shape=(3, 2, 4), dtype=DType("float32"), device=Device("cpu")))

    def mock_einsum(*args):
        return "es"

    orig_einsum = linear.einsum
    orig_add2 = linear.add
    linear.einsum = mock_einsum
    linear.add = mock_add
    try:
        assert linear.bilinear(t_in1, t_in2, w_bi) == "es"
        assert linear.bilinear(t_in1, t_in2, w_bi, b) == "add"
    finally:
        linear.einsum = orig_einsum
        linear.add = orig_add2
