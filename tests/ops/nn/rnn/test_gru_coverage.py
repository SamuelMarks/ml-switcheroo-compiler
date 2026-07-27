# ruff: noqa: E501
import sys

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig

gru_mod = sys.modules["ml_switcheroo_compiler.ops.nn.gru"]


def test_gru_coverage():
    config.eager_mode = True
    t_in = Tensor(np.ones((1, 2)), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    t_state = Tensor(np.ones((1, 4)), TensorConfig(shape=(1, 4), dtype=DType("float32"), device=Device("cpu")))
    w = Tensor(np.ones((2, 12)), TensorConfig(shape=(2, 12), dtype=DType("float32"), device=Device("cpu")))
    rw = Tensor(np.ones((4, 12)), TensorConfig(shape=(4, 12), dtype=DType("float32"), device=Device("cpu")))
    b = Tensor(np.ones((12,)), TensorConfig(shape=(12,), dtype=DType("float32"), device=Device("cpu")))

    def mock_matmul(*args):
        return Tensor(np.ones((1, 12)), TensorConfig(shape=(1, 12), dtype=DType("float32"), device=Device("cpu")))

    def mock_gates(*args):
        return "hidden"

    orig_matmul = gru_mod.matmul
    orig_gates = gru_mod._compute_gru_gates
    gru_mod.matmul = mock_matmul
    gru_mod._compute_gru_gates = mock_gates
    try:
        res = gru_mod.gru_cell(t_in, t_state, w, rw, b)
        assert res == ("hidden", "hidden")
        res = gru_mod.gru_cell(t_in, t_state, w, rw)
        assert res == ("hidden", "hidden")
    finally:
        gru_mod.matmul = orig_matmul
        gru_mod._compute_gru_gates = orig_gates

    assert gru_mod.Gru().infer_shape(t_in) is t_in
    assert gru_mod.Gru().infer_shape() == ()

    class MockBackend:
        def execute_op(self, *args, **kwargs):
            return "gru_res"

    orig_backend = gru_mod.get_active_backend
    gru_mod.get_active_backend = lambda: MockBackend()
    try:
        assert gru_mod.gru(t_in) == "gru_res"
    finally:
        gru_mod.get_active_backend = orig_backend

    config.eager_mode = False
    orig_emit = gru_mod._emit_shape_node

    def dummy_emit(*args, **kwargs):
        return "emitted"

    gru_mod._emit_shape_node = dummy_emit
    try:
        assert gru_mod.gru(t_in) == "emitted"
    finally:
        gru_mod._emit_shape_node = orig_emit
        config.eager_mode = True
