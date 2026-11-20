"""Behavioral tests for GRU cell arithmetic, shape inference, eager dispatch, and IR tracing."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.gru import Gru, _sigmoid, gru, gru_cell
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, TracerTape
from tests.testing_utils import ensure_tensor


def test_gru_cell_sigmoid_and_shape_inference() -> None:
    """Verify `gru_cell` gate computation (with and without bias), `_sigmoid`, and `Gru.infer_shape`."""
    with ConfigContext(backend="numpy", eager_mode=True):
        x = Tensor(np.zeros((2, 3), dtype=np.float32), TensorConfig((2, 3), DType.Float32, Device("cpu")))
        state = Tensor(np.zeros((2, 2), dtype=np.float32), TensorConfig((2, 2), DType.Float32, Device("cpu")))
        w = Tensor(np.zeros((3, 6), dtype=np.float32), TensorConfig((3, 6), DType.Float32, Device("cpu")))
        rw = Tensor(np.zeros((2, 6), dtype=np.float32), TensorConfig((2, 6), DType.Float32, Device("cpu")))
        b = Tensor(np.zeros((6,), dtype=np.float32), TensorConfig((6,), DType.Float32, Device("cpu")))

        res_bias = gru_cell(x, state, w, rw, b)
        assert isinstance(res_bias, tuple)
        out_bias = ensure_tensor(res_bias[0])
        st_bias = ensure_tensor(res_bias[1])
        assert out_bias.shape == (2, 2)
        np.testing.assert_allclose(np.asarray(out_bias.data), np.zeros((2, 2), dtype=np.float32))
        assert st_bias is out_bias

        res_nobias = gru_cell(x, state, w, rw)
        assert isinstance(res_nobias, tuple)
        out_nobias = ensure_tensor(res_nobias[0])
        st_nobias = ensure_tensor(res_nobias[1])
        assert out_nobias.shape == (2, 2)
        np.testing.assert_allclose(np.asarray(out_nobias.data), np.zeros((2, 2), dtype=np.float32))
        assert st_nobias is out_nobias

        sig_zero = _sigmoid(x)
        np.testing.assert_allclose(np.asarray(sig_zero.data), np.full((2, 3), 0.5, dtype=np.float32))

    gru_op = Gru()
    assert gru_op.infer_shape(x) is x
    assert gru_op.infer_shape() == ()


def test_gru_eager_and_symbolic_tracing_ir_emission() -> None:
    """Verify `gru` dispatches eagerly and emits canonical `Gru` `LogicalNode`s when traced."""
    t_in = Tensor(np.ones((2, 3), dtype=np.float32), TensorConfig((2, 3), DType.Float32, Device("cpu")))

    gru_module = importlib.import_module("ml_switcheroo_compiler.ops.nn.gru")
    backend = MagicMock()
    backend.execute_op.return_value = np.ones((2, 3), dtype=np.float32)
    with (
        ConfigContext(eager_mode=True),
        patch.object(gru_module, "get_active_backend", return_value=backend),
    ):
        eager_res = gru(t_in)
        np.testing.assert_allclose(np.asarray(eager_res), np.ones((2, 3), dtype=np.float32))
        assert backend.execute_op.call_count == 1
        assert backend.execute_op.call_args[0][0] == "Gru"
        assert backend.execute_op.call_args[0][1] is t_in.data

    t_sym = Tensor(
        ProxyTensor("gru_in", (2, 3), "float64"),
        TensorConfig((2, 3), DType.Float64, Device("cpu")),
    )
    with ConfigContext(eager_mode=False):
        tape = TracerTape()
        traced_graph = tape.start_tracing("gru_trace")
        try:
            out_tensor = gru(t_sym)
            out_scalar = gru("raw_non_tensor")
        finally:
            tape.stop_tracing()

    assert isinstance(out_tensor, Tensor)
    assert out_tensor.shape == (2, 3)
    assert out_tensor.dtype == DType.Float64

    out_scalar_t = ensure_tensor(out_scalar)
    assert out_scalar_t.shape == ()
    assert out_scalar_t.dtype == DType.Float32

    gru_nodes = [n for n in traced_graph.nodes.values() if n.op_type == "Gru"]
    assert len(gru_nodes) == 2
    assert any(n.inputs == ["gru_in"] and n.shape_metadata == (2, 3) for n in gru_nodes)
