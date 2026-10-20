"""Tests for test_quantized_mixed_precision."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def _test_quantized_ops_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_quantized_ops.

    Args:
        _state: Shared test state dictionary.
    """
    import numpy as np

    import ml_switcheroo_compiler.ops.nn.quantized_ops as q_ops
    import ml_switcheroo_compiler.tracing.state as state

    orig = config.eager_mode
    orig_tracing = state.global_tracing_state.is_tracing
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    setattr(state.global_tracing_state, "add_node", lambda node: None)
    getattr(_state, "update")({"np": np, "orig": orig, "orig_add_node": orig_add_node, "orig_tracing": orig_tracing})
    getattr(_state, "update")({"q_ops": q_ops, "state": state})
    _state.update(locals())


def _test_quantized_ops_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_quantized_ops.

    Args:
        _state: Shared test state dictionary.
    """
    np, orig, orig_add_node, orig_tracing = _state["np"], _state["orig"], _state["orig_add_node"], _state["orig_tracing"]
    q_ops, state = _state["q_ops"], _state["state"]
    try:
        t = Tensor(np.array([1.0]), TensorConfig(shape=(1,), dtype=DType("float32"), device=Device("cpu")))

        q_ops.fake_quant_with_min_max_vars(t, t, t)

        qp = q_ops.QuantizationParams()
        q_ops.quantize_and_dequantize(t, t, t, qp)

        q_ops.abs_max_quantize(t)

        q_ops.compute_float8_amax_history(t, t)
        q_ops.compute_float8_scale(t, t)

        q_ops.quantize(t)

        qc = q_ops.QuantizationConfig()
        qoc = q_ops.QuantizedOpsConfig(weight=t, scales=t, biases=t, indices=t, q_config=qc)

        q_ops.quantized_matmul(t, qoc)
        q_ops.gather_qmm(t, qoc)
        q_ops.quantized_linear(t, qoc)
        q_ops.quantized_embedding(t, qoc)
        q_ops.dequantize(t, t, t)
        q_ops.quantized_conv(t, qoc)

        # Test eagerness
        config.eager_mode = True

        from unittest.mock import patch

        with patch("ml_switcheroo_compiler.ops.nn.linear_ops.linear", return_value=t):
            with patch("ml_switcheroo_compiler.ops.creation.frontend_basic.zeros_like", return_value=t):
                q_ops.fake_quant_with_min_max_vars(t, t, t)
                q_ops.quantize_and_dequantize(t, t, t, qp)
                q_ops.abs_max_quantize(t)
                q_ops.compute_float8_amax_history(t, t)
                q_ops.compute_float8_scale(t, t)
                q_ops.quantize(t)
                q_ops.quantized_matmul(t, qoc)
                q_ops.gather_qmm(t, qoc)

        class DummyShape:
            """Test class DummyShape."""

            shape = (1, 2)

        assert q_ops.QuantizeOp().infer_shape(DummyShape()) == (1, 2)
        assert q_ops.QuantizedMatmulOp().infer_shape(DummyShape()) == (1, 2)
        assert q_ops.GatherQMMOp().infer_shape(DummyShape()) == (1, 2)
        assert q_ops.FakeQuantWithMinMaxVarsOp().infer_shape(DummyShape()) == (1, 2)
        assert q_ops.QuantizeAndDequantizeOp().infer_shape(DummyShape()) == (1, 2)
        assert q_ops.AbsMaxQuantizeOp().infer_shape(DummyShape()) == (1, 2)
        assert q_ops.ComputeFloat8AmaxHistoryOp().infer_shape(DummyShape()) == (1, 2)
        assert q_ops.ComputeFloat8ScaleOp().infer_shape(DummyShape()) == (1, 2)

    finally:
        setattr(config, "eager_mode", orig)
        setattr(state.global_tracing_state, "is_tracing", orig_tracing)
        setattr(state.global_tracing_state, "add_node", orig_add_node)


def test_quantized_ops() -> None:
    """Test function test_quantized_ops."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_quantized_ops_part_1(_state)
    _test_quantized_ops_part_2(_state)
    assert len(_state) > 0
