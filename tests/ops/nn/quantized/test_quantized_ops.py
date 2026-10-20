"""Unit tests for quantized neural network operations."""

from __future__ import annotations

import typing
from unittest.mock import patch

import numpy as np

import ml_switcheroo_compiler.ops.nn.quantized_ops as q_ops
from ml_switcheroo_compiler.core.config import ConfigContext, config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.quantized_ops import (
    AbsMaxQuantizeOp,
    ComputeFloat8AmaxHistoryOp,
    ComputeFloat8ScaleOp,
    FakeQuantWithMinMaxVarsOp,
    GatherQMMOp,
    QuantizeAndDequantizeOp,
    QuantizedMatmulOp,
    QuantizedOpsConfig,
    QuantizeOp,
    abs_max_quantize,
    compute_float8_amax_history,
    compute_float8_scale,
    dequantize,
    fake_quant_with_min_max_vars,
    gather_qmm,
    quantize,
    quantize_and_dequantize,
    quantized_conv,
    quantized_embedding,
    quantized_linear,
    quantized_matmul,
)
from ml_switcheroo_compiler.tracing.state import global_tracing_state

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_quantized_ops_trace() -> None:
    """Test symbolic tracing and shape inference for all quantized operations."""
    orig = config.eager_mode
    config.eager_mode = False

    try:
        t = Tensor(np.array([[1.0, 2.0]]), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))

        with (
            patch("ml_switcheroo_compiler.ops.nn.quantized_ops._emit_shape_node", return_value=t),
            patch("ml_switcheroo_compiler.ops.nn.quantized_ops.linear", return_value=t),
            patch("ml_switcheroo_compiler.ops.nn.quantized_ops.gather", return_value=t),
        ):
            assert fake_quant_with_min_max_vars(t, t, t) is t

            qp = q_ops.QuantizationParams()
            assert quantize_and_dequantize(t, t, t, qp) is t
            assert abs_max_quantize(t) == (t, t)
            assert compute_float8_amax_history(t, t) is t
            assert compute_float8_scale(t, t) is t
            assert quantize(t) == (t, t, t)

            qc = q_ops.QuantizationConfig()
            qoc = q_ops.QuantizedOpsConfig(weight=t, scales=t, biases=t, indices=t, q_config=qc)

            assert quantized_matmul(t, qoc) is t
            assert gather_qmm(t, qoc) is t

            for op_class in [
                QuantizeOp,
                QuantizedMatmulOp,
                GatherQMMOp,
                FakeQuantWithMinMaxVarsOp,
                QuantizeAndDequantizeOp,
                AbsMaxQuantizeOp,
                ComputeFloat8AmaxHistoryOp,
                ComputeFloat8ScaleOp,
            ]:
                op = op_class()
                assert op.infer_shape(t) == (1, 2)

            assert quantized_linear(t, qoc) is not None
            assert quantized_embedding(t, qoc) is not None
            assert dequantize(t, t, t) is not None
            assert quantized_conv(t, qoc) is not None

            qoc_no_biases = q_ops.QuantizedOpsConfig(weight=t, scales=t, biases=None, indices=None, q_config=None)
            assert quantized_matmul(t, qoc_no_biases) is t
            assert gather_qmm(t, qoc_no_biases) is t
            assert quantized_linear(t, qoc_no_biases) is not None
            assert dequantize(t, t, biases=None) is not None
            assert quantized_conv(t, qoc_no_biases) is not None
            assert quantize_and_dequantize(t, t, t, params=None) is t
    finally:
        config.eager_mode = orig


def create_eager_tensor(data: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> Tensor:
    """Create a CPU Float32 Tensor from a NumPy array.

    Args:
        data (np.ndarray): Input array.

    Returns:
        Tensor: Wrapped tensor.
    """
    return Tensor(data, TensorConfig(data.shape, DType.Float32, Device("cpu")))


def test_quantized_ops_missing() -> None:
    """Test eager quantized_conv execution."""
    inp = create_eager_tensor(np.ones((1, 2, 2, 1)))
    weight = create_eager_tensor(np.ones((2, 2, 1, 1)))
    scales = create_eager_tensor(np.ones((1,)))
    q_cfg = QuantizedOpsConfig(weight=weight, scales=scales, biases=None)

    with ConfigContext(eager_mode=True):
        res = quantized_conv(inp, q_cfg)
        assert isinstance(res, Tensor)
        typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(typing.cast("EagerCallableProtocol", res.numpy)(), typing.cast("EagerCallableProtocol", np.array)([[[[4.0]]]]))


def test_quantized_ops_brute() -> None:
    """Test eager and tracing execution of quantized operations."""
    config.backend = "numpy"
    config.eager_mode = True

    t = Tensor(np.ones((4, 4), dtype=np.float32), TensorConfig((4, 4), "float32", "cpu"))
    s = Tensor(np.ones((1,), dtype=np.float32), TensorConfig((1,), "float32", "cpu"))
    idx = Tensor(np.array([0, 1], dtype=np.int32), TensorConfig((2,), "int32", "cpu"))

    assert fake_quant_with_min_max_vars(t, s, s) is not None
    assert quantize_and_dequantize(t, s, s) is not None
    assert abs_max_quantize(t) is not None
    assert compute_float8_amax_history(t, t) is not None
    assert compute_float8_scale(t, t) is not None
    assert quantize(t) is not None

    qconf = QuantizedOpsConfig(weight=t, scales=t, biases=t, indices=idx)
    assert quantized_matmul(t, qconf) is not None
    assert gather_qmm(t, qconf) is not None
    assert quantized_linear(t, qconf) is not None
    assert quantized_embedding(idx, qconf) is not None
    assert dequantize(t, t, t) is not None

    config.eager_mode = False
    global_tracing_state.is_tracing = True
    try:
        with patch("ml_switcheroo_compiler.ops.nn.quantized_ops._emit_shape_node", return_value=t):
            assert fake_quant_with_min_max_vars(t, s, s) is t
            assert quantize_and_dequantize(t, s, s) is t
            assert abs_max_quantize(t) == (t, t)
            assert compute_float8_amax_history(t, t) is t
            assert compute_float8_scale(t, t) is t
            assert quantize(t) == (t, t, t)
            assert quantized_matmul(t, qconf) is t
            assert gather_qmm(t, qconf) is t
            assert QuantizeOp().infer_shape(t) == (4, 4)
            assert QuantizedMatmulOp().infer_shape(t) == (4, 4)
            assert GatherQMMOp().infer_shape(t) == (4, 4)
            assert FakeQuantWithMinMaxVarsOp().infer_shape(t) == (4, 4)
            assert QuantizeAndDequantizeOp().infer_shape(t) == (4, 4)
            assert AbsMaxQuantizeOp().infer_shape(t) == (4, 4)
            assert ComputeFloat8AmaxHistoryOp().infer_shape(t) == (4, 4)
            assert ComputeFloat8ScaleOp().infer_shape(t) == (4, 4)
    finally:
        global_tracing_state.is_tracing = False
