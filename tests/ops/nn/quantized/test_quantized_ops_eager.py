"""Tests for test_quantized_ops_eager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def _test_quantized_ops_eager_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_quantized_ops_eager.

    Args:
        _state: Shared test state dictionary.
    """
    np, q_ops = _state["np"], _state["q_ops"]
    t = Tensor(np.array([[1.0, 2.0]]), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))

    q_ops.fake_quant_with_min_max_vars(t, t, t)

    qp = q_ops.QuantizationParams()
    q_ops.quantize_and_dequantize(t, t, t, qp)

    q_ops.abs_max_quantize(t)

    q_ops.compute_float8_amax_history(t, t)
    q_ops.compute_float8_scale(t, t)

    q_ops.quantize(t)

    qc = q_ops.QuantizationConfig()
    qoc = q_ops.QuantizedOpsConfig(weight=t, scales=t, biases=t, indices=t, q_config=qc)

    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.nn.linear_ops.linear", return_value=t):
        q_ops.quantized_matmul(t, qoc)
        q_ops.gather_qmm(t, qoc)

    # Test actual eager dequantization mathematical accuracy
    w_quant = Tensor(np.array([[2.0, 4.0]]), TensorConfig((1, 2), "float32", "cpu"))
    scales = Tensor(np.array([[0.5, 0.5]]), TensorConfig((1, 2), "float32", "cpu"))
    zeros = Tensor(np.array([[1.0, 1.0]]), TensorConfig((1, 2), "float32", "cpu"))

    # W_float = (W_quant - zeros) * scales = ([[2, 4]] - [[1, 1]]) * [[0.5, 0.5]] = [[1, 3]] * [[0.5, 0.5]] = [[0.5, 1.5]]
    qoc_math = q_ops.QuantizedOpsConfig(weight=w_quant, scales=scales, zeros=zeros, biases=None, indices=None, q_config=None)
    getattr(_state, "update")({"patch": patch, "qc": qc, "qoc": qoc, "qoc_math": qoc_math})
    getattr(_state, "update")({"qp": qp, "scales": scales, "t": t, "w_quant": w_quant})
    getattr(_state, "update")({"zeros": zeros})
    _state.update(locals())


def _test_quantized_ops_eager_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_quantized_ops_eager.

    Args:
        _state: Shared test state dictionary.
    """
    np, q_ops, qoc_math, scales = _state["np"], _state["q_ops"], _state["qoc_math"], _state["scales"]
    w_quant, zeros = _state["w_quant"], _state["zeros"]
    indices = Tensor(np.array([[0]]), TensorConfig((1, 1), "int32", "cpu"))
    embed_res = q_ops.quantized_embedding(indices, qoc_math)
    assert np.allclose(embed_res.data, [[0.5, 1.5]])

    # Test quantized_linear math (input of ones)
    inp = Tensor(np.array([[1.0, 1.0]]), TensorConfig((1, 2), "float32", "cpu"))
    # we expect input @ weight_float.T = [[1.0, 1.0]] @ [[0.5, 1.5]].T = 0.5 * 1.0 + 1.5 * 1.0 = 2.0
    lin_res = q_ops.quantized_linear(inp, qoc_math)
    assert np.allclose(lin_res.data, [[2.0]])

    # Test dequantize mathematical correctness
    deq_res = q_ops.dequantize(w_quant, scales, zeros)
    assert np.allclose(deq_res.data, [[0.5, 1.5]])

    # Coverage for zeros=None, biases=None in eager mode
    qoc_none = q_ops.QuantizedOpsConfig(weight=w_quant, scales=scales, zeros=None, biases=None, indices=None, q_config=None)
    embed_none = q_ops.quantized_embedding(indices, qoc_none)
    assert embed_none is not None
    lin_none = q_ops.quantized_linear(inp, qoc_none)
    assert lin_none is not None
    deq_none = q_ops.dequantize(w_quant, scales, biases=None)
    assert deq_none is not None

    # Test quantized_conv 2D mathematical correctness
    input_2d_val = np.ones((1, 3, 3, 1), dtype=np.float32)
    input_2d = Tensor(input_2d_val, TensorConfig((1, 3, 3, 1), DType.Float32, Device("cpu")))

    w_2d_quant = Tensor(np.array([[[[2.0]], [[2.0]]], [[[2.0]], [[2.0]]]]), TensorConfig((2, 2, 1, 1), DType.Float32, Device("cpu")))
    getattr(_state, "update")({"deq_none": deq_none, "deq_res": deq_res, "embed_none": embed_none, "embed_res": embed_res})
    getattr(_state, "update")({"indices": indices, "inp": inp, "input_2d": input_2d, "input_2d_val": input_2d_val})
    getattr(_state, "update")({"lin_none": lin_none, "lin_res": lin_res, "qoc_none": qoc_none, "w_2d_quant": w_2d_quant})
    _state.update(locals())


def _test_quantized_ops_eager_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 3 of test_quantized_ops_eager.

    Args:
        _state: Shared test state dictionary.
    """
    input_2d, np, q_ops, w_2d_quant = _state["input_2d"], _state["np"], _state["q_ops"], _state["w_2d_quant"]
    scales_2d = Tensor(np.ones((2, 2, 1, 1), dtype=np.float32) * 0.5, TensorConfig((2, 2, 1, 1), DType.Float32, Device("cpu")))
    zeros_2d = Tensor(np.ones((2, 2, 1, 1), dtype=np.float32) * 1.0, TensorConfig((2, 2, 1, 1), DType.Float32, Device("cpu")))

    # dequantized weight will be ones * 0.5
    # Convolution of a 3x3 input of ones with 2x2 kernel of 0.5s results in a 2x2 output of:
    # 0.5 * 1.0 * 4 = 2.0 per output patch (under VALID padding)
    qoc_conv = q_ops.QuantizedOpsConfig(weight=w_2d_quant, scales=scales_2d, biases=zeros_2d, indices=None, q_config=None)

    conv_res = q_ops.quantized_conv(input_2d, qoc_conv, stride=1, padding="VALID")
    assert np.allclose(conv_res.data, [[[[2.0]], [[2.0]]], [[[2.0]], [[2.0]]]])

    # Test fake_quantize_per_channel_affine
    input_q_chan = Tensor(np.array([[[0.1, -0.1], [0.8, -0.8]]]), TensorConfig((1, 2, 2), DType.Float32, Device("cpu")))
    scale_q_chan = Tensor(np.array([0.1, 0.2]), TensorConfig((2,), DType.Float32, Device("cpu")))
    zero_point_q_chan = Tensor(np.array([0.0, 0.0]), TensorConfig((2,), DType.Int32, Device("cpu")))

    fq_chan_res = q_ops.fake_quantize_per_channel_affine(input_q_chan, scale_q_chan, zero_point_q_chan, axis=2, quant_min=-128, quant_max=127)
    assert fq_chan_res is not None

    # Test fake_quantize_per_tensor_affine
    input_q_tens = Tensor(np.array([0.1, -0.1, 0.8, -0.8]), TensorConfig((4,), DType.Float32, Device("cpu")))
    fq_tens_res = q_ops.fake_quantize_per_tensor_affine(input_q_tens, scale=0.1, zero_point=0, quant_min=-128, quant_max=127)
    assert fq_tens_res is not None
    getattr(_state, "update")({"conv_res": conv_res, "fq_chan_res": fq_chan_res, "fq_tens_res": fq_tens_res, "input_q_chan": input_q_chan})
    getattr(_state, "update")({"input_q_tens": input_q_tens, "qoc_conv": qoc_conv, "scale_q_chan": scale_q_chan, "scales_2d": scales_2d})
    getattr(_state, "update")({"zero_point_q_chan": zero_point_q_chan, "zeros_2d": zeros_2d})
    _state.update(locals())


def test_quantized_ops_eager() -> None:
    """Test function test_quantized_ops_eager."""
    import numpy as np

    import ml_switcheroo_compiler.ops.nn.quantized_ops as q_ops

    orig = config.eager_mode
    config.eager_mode = True

    try:
        _state: dict[str, EagerTensorProtocol] = dict(locals())
        getattr(_state, "update")({"np": np, "orig": orig, "q_ops": q_ops})
        for _sub_fn in (
            _test_quantized_ops_eager_sub_1,
            _test_quantized_ops_eager_sub_2,
            _test_quantized_ops_eager_sub_3,
        ):
            _sub_fn(_state)
        assert isinstance(_state, dict)

    finally:
        config.eager_mode = orig
