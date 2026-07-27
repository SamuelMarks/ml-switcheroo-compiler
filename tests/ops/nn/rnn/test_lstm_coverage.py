# ruff: noqa: E501
import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.lstm import lstm_cell


def test_lstm_coverage():
    config.eager_mode = True
    t_in = Tensor(np.array([[1.0, 2.0]]), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    t_h = Tensor(np.array([[1.0, 1.0]]), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    t_c = Tensor(np.array([[1.0, 1.0]]), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    w = Tensor(np.array([[1.0] * 8] * 2), TensorConfig(shape=(2, 8), dtype=DType("float32"), device=Device("cpu")))
    rw = Tensor(np.array([[1.0] * 8] * 2), TensorConfig(shape=(2, 8), dtype=DType("float32"), device=Device("cpu")))
    b = Tensor(np.array([1.0] * 8), TensorConfig(shape=(8,), dtype=DType("float32"), device=Device("cpu")))

    res_h, res_c = lstm_cell(t_in, (t_h, t_c), w, rw, b)
    assert res_h is not None and res_c is not None
    res_h, res_c = lstm_cell(t_in, (t_h, t_c), w, rw)
    assert res_h is not None and res_c is not None
