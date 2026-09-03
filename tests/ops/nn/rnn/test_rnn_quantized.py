# ruff: noqa: E501
from unittest.mock import patch

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.quantized_ops import (
    AbsMaxQuantizeOp,
    ComputeFloat8AmaxHistoryOp,
    ComputeFloat8ScaleOp,
    FakeQuantWithMinMaxVarsOp,
    GatherQMMOp,
    QuantizationConfig,
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
from ml_switcheroo_compiler.ops.nn.rnn_cell import simple_rnn_cell
from ml_switcheroo_compiler.ops.nn.rnn_utils import BidirectionalConfig, BidirectionalInputs, DropoutWrapperConfig, RNNCellDeviceWrapper, RNNCellDropoutWrapper, RNNCellResidualWrapper, RNNConfig, ScanConfig, bidirectional, rnn, scan
from ml_switcheroo_compiler.ops.nn.time_distributed import TimeDistributed, time_distributed


def test_quantized_coverage():
    config.eager_mode = True
    t = Tensor(np.array([[1.0, 2.0]]), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    t_min = Tensor(np.array([0.0]), TensorConfig(shape=(1,), dtype=DType("float32"), device=Device("cpu")))
    t_max = Tensor(np.array([1.0]), TensorConfig(shape=(1,), dtype=DType("float32"), device=Device("cpu")))

    # Fake quant
    assert fake_quant_with_min_max_vars(t, t_min, t_max) is not None
    assert fake_quant_with_min_max_vars(t, t_min, t_max, narrow_range=True) is not None

    # Quantize and dequantize
    assert quantize_and_dequantize(t, t_min, t_max) is not None

    # AbsMaxQuantize
    q, s = abs_max_quantize(t)
    assert q is not None and s is not None

    # Float8
    assert compute_float8_amax_history(t, t) is not None
    assert compute_float8_scale(t, t) is not None

    # Quantize
    qw, qs, qb = quantize(t)
    assert qw is not None and qs is not None and qb is not None

    q_conf = QuantizedOpsConfig(weight=t, scales=t, biases=t, indices=t, q_config=QuantizationConfig(transpose=False))
    q_conf_t = QuantizedOpsConfig(weight=t, scales=t, biases=t, indices=t, q_config=QuantizationConfig(transpose=True))
    assert quantized_matmul(t, q_conf) is not None
    assert gather_qmm(t, q_conf) is not None
    assert quantized_linear(t, q_conf) is not None

    t_idx = Tensor(np.array([[0, 0]]), TensorConfig(shape=(1, 2), dtype=DType("int32"), device=Device("cpu")))
    assert quantized_embedding(t_idx, q_conf) is not None

    # Tracing
    original_eager = config.eager_mode
    try:
        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

        class DummyGraph:
            name = "dummy"
            nodes = {}

            def add_node(self, node):
                pass

        global_tracing_state.active_graph = DummyGraph()

        assert fake_quant_with_min_max_vars(t, t_min, t_max) is not None
        assert quantize_and_dequantize(t, t_min, t_max) is not None
        q, s = abs_max_quantize(t)
        assert q is not None and s is not None
        assert compute_float8_amax_history(t, t) is not None
        assert compute_float8_scale(t, t) is not None
        qw, qs, qb = quantize(t)
        assert qw is not None and qs is not None and qb is not None

        assert quantized_matmul(t, q_conf) is not None
        assert quantized_matmul(t, q_conf_t) is not None
        assert gather_qmm(t, q_conf) is not None
        assert gather_qmm(t, q_conf_t) is not None

        assert dequantize(t, t, t) is not None
        assert quantized_conv(t, q_conf) is not None
    finally:
        config.eager_mode = original_eager
        global_tracing_state.is_tracing = False

    class DummyShape:
        shape = (1, 2)

    assert QuantizeOp().infer_shape(DummyShape()) == (1, 2)
    assert QuantizedMatmulOp().infer_shape(DummyShape()) == (1, 2)
    assert GatherQMMOp().infer_shape(DummyShape()) == (1, 2)
    assert FakeQuantWithMinMaxVarsOp().infer_shape(DummyShape()) == (1, 2)
    assert QuantizeAndDequantizeOp().infer_shape(DummyShape()) == (1, 2)
    assert AbsMaxQuantizeOp().infer_shape(DummyShape()) == (1, 2)
    assert ComputeFloat8AmaxHistoryOp().infer_shape(DummyShape()) == (1, 2)
    assert ComputeFloat8ScaleOp().infer_shape(DummyShape()) == (1, 2)


def test_rnn_coverage():
    config.eager_mode = True
    t_in = Tensor(np.ones((2, 1, 4)), TensorConfig(shape=(2, 1, 4), dtype=DType("float32"), device=Device("cpu")))  # seq_len, batch, dim
    t_h = Tensor(np.ones((1, 4)), TensorConfig(shape=(1, 4), dtype=DType("float32"), device=Device("cpu")))
    w = Tensor(np.ones((4, 4)), TensorConfig(shape=(4, 4), dtype=DType("float32"), device=Device("cpu")))
    rw = Tensor(np.ones((4, 4)), TensorConfig(shape=(4, 4), dtype=DType("float32"), device=Device("cpu")))
    b = Tensor(np.ones((4,)), TensorConfig(shape=(4,), dtype=DType("float32"), device=Device("cpu")))

    assert simple_rnn_cell(t_in[0], (t_h,), w, rw, b) is not None

    def dummy_cell(inputs, state):
        return simple_rnn_cell(inputs, state, w, rw, b)

    out, final_state = rnn(t_in, (t_h,), dummy_cell, RNNConfig(time_major=True))
    assert out is not None and final_state is not None

    t_in_batch_major = Tensor(np.ones((1, 2, 4)), TensorConfig(shape=(1, 2, 4), dtype=DType("float32"), device=Device("cpu")))
    out, final_state = rnn(t_in_batch_major, (t_h,), dummy_cell, RNNConfig(time_major=False))
    assert out is not None and final_state is not None

    # Bidirectional
    bidi_in = BidirectionalInputs(t_in, t_in, (t_h,), (t_h,))
    out, fwd, bwd = bidirectional(bidi_in, dummy_cell, BidirectionalConfig(merge_mode="concat", time_major=True))
    assert out is not None
    out, fwd, bwd = bidirectional(bidi_in, dummy_cell, BidirectionalConfig(merge_mode="sum", time_major=True))
    assert out is not None
    out, fwd, bwd = bidirectional(bidi_in, dummy_cell, BidirectionalConfig(merge_mode="mul", time_major=True))
    assert out is not None
    out, fwd, bwd = bidirectional(bidi_in, dummy_cell, BidirectionalConfig(merge_mode="ave", time_major=True))
    assert out is not None
    out, fwd, bwd = bidirectional(bidi_in, dummy_cell, BidirectionalConfig(merge_mode="none", time_major=True))
    assert out is not None

    # scan eager unroll
    def scan_dummy(carry, x):
        return dummy_cell(x, carry)

    carry, ys = scan(scan_dummy, (t_h,), t_in, config=ScanConfig(unroll=True, reverse=True))
    assert carry is not None and ys is not None

    original_eager = config.eager_mode
    try:
        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

        class DummyGraph:
            name = "dummy"
            nodes = {}

            def add_node(self, node):
                pass

        global_tracing_state.active_graph = DummyGraph()

        # Needs a patch for cf_scan probably, we just want to execute the scan function body
        # actually scan with config.eager_mode = False calls cf_scan
        with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.cf_scan") as mock_cf_scan:
            mock_cf_scan.return_value = ("carry", "y")
            carry, ys = scan(scan_dummy, (t_h,), t_in, config=ScanConfig(unroll=False, reverse=True))
            assert carry == "carry" and ys == "y"
            carry, ys = scan(scan_dummy, (t_h,), t_in, config=ScanConfig(unroll=False, reverse=False))
            assert carry == "carry" and ys == "y"
    finally:
        config.eager_mode = original_eager
        global_tracing_state.is_tracing = False

    # Wrappers
    dev_wrapper = RNNCellDeviceWrapper(dummy_cell, Device("cpu"))
    assert dev_wrapper(t_in[0], (t_h,)) is not None

    do_wrapper = RNNCellDropoutWrapper(dummy_cell, DropoutWrapperConfig(input_keep_prob=0.5, output_keep_prob=0.5))
    with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.dropout") as mock_do:
        mock_do.return_value = t_in[0]
        assert do_wrapper(t_in[0], (t_h,)) is not None

    res_wrapper = RNNCellResidualWrapper(dummy_cell)
    assert res_wrapper(t_in[0], (t_h,)) is not None

    res_wrapper2 = RNNCellResidualWrapper(dummy_cell, residual_fn=lambda x, y: y)
    assert res_wrapper2(t_in[0], (t_h,)) is not None


def test_time_distributed_coverage():
    config.eager_mode = True
    t = Tensor(np.ones((2, 2)), TensorConfig(shape=(2, 2), dtype=DType("float32"), device=Device("cpu")))

    assert TimeDistributed().infer_shape(t) == (2, 2)

    with patch("ml_switcheroo_compiler.ops.nn.time_distributed.get_op") as mock_get_op:
        mock_op = mock_get_op.return_value.return_value
        mock_op.return_value = "timed"
        with pytest.raises(Exception):
            time_distributed(t, "Dense")
