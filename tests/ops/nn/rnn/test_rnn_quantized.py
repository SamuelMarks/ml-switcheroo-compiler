"""Tests for test_rnn_quantized."""

from __future__ import annotations

import typing
from unittest.mock import patch

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
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
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable, ensure_shape, ensure_tensor_dict

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def _test_quantized_coverage_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_quantized_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    config.eager_mode = True
    t = Tensor(np.array([[1.0, 2.0]]), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    t_min = Tensor(np.array([0.0]), TensorConfig(shape=(1,), dtype=DType("float32"), device=Device("cpu")))
    t_max = Tensor(np.array([1.0]), TensorConfig(shape=(1,), dtype=DType("float32"), device=Device("cpu")))
    assert fake_quant_with_min_max_vars(t, t_min, t_max) is not None
    assert fake_quant_with_min_max_vars(t, t_min, t_max, narrow_range=True) is not None
    assert quantize_and_dequantize(t, t_min, t_max) is not None
    (q, s) = ensure_callable(abs_max_quantize)(t)
    assert ensure_eager_tensor(q) is not None and s is not None
    assert compute_float8_amax_history(t, t) is not None
    assert compute_float8_scale(t, t) is not None
    (qw, qs, qb) = ensure_callable(quantize)(t)
    assert ensure_eager_tensor(qw) is not None and ensure_eager_tensor(qs) is not None and (qb is not None)
    q_conf = QuantizedOpsConfig(weight=t, scales=t, biases=t, indices=t, q_config=QuantizationConfig(transpose=False))
    q_conf_t = QuantizedOpsConfig(weight=t, scales=t, biases=t, indices=t, q_config=QuantizationConfig(transpose=True))
    assert quantized_matmul(t, q_conf) is not None
    assert gather_qmm(t, q_conf) is not None
    assert quantized_linear(t, q_conf) is not None
    t_idx = Tensor(np.array([[0, 0]]), TensorConfig(shape=(1, 2), dtype=DType("int32"), device=Device("cpu")))
    assert quantized_embedding(t_idx, q_conf) is not None
    original_eager = config.eager_mode
    _state.update(ensure_tensor_dict({"original_eager": original_eager, "q": q, "q_conf": q_conf, "q_conf_t": q_conf_t}))
    _state.update(ensure_tensor_dict({"qb": qb, "qs": qs, "qw": qw, "s": s}))
    _state.update(ensure_tensor_dict({"t": t, "t_idx": t_idx, "t_max": t_max, "t_min": t_min}))


def _test_quantized_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_quantized_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (original_eager, q_conf, q_conf_t, t) = (_state["original_eager"], _state["q_conf"], _state["q_conf_t"], _state["t"])
    (t_max, t_min) = (_state["t_max"], _state["t_min"])
    try:
        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

        class DummyGraph:
            """Test class DummyGraph."""

            name = "dummy"
            nodes = ensure_tensor_dict({})

            def add_node(self, node: EagerValue) -> None:
                """Test function add_node.

                Args:
                    node: Test parameter `node`.
                """
                pass

        ensure_mutable(global_tracing_state).active_graph = ensure_eager_tensor(DummyGraph())
        assert ensure_callable(fake_quant_with_min_max_vars)(t, t_min, t_max) is not None
        assert ensure_callable(quantize_and_dequantize)(t, t_min, t_max) is not None
        (q, s) = ensure_callable(abs_max_quantize)(t)
        assert ensure_eager_tensor(q) is not None and s is not None
        assert ensure_callable(compute_float8_amax_history)(t, t) is not None
        assert ensure_callable(compute_float8_scale)(t, t) is not None
        (qw, qs, qb) = ensure_callable(quantize)(t)
        assert ensure_eager_tensor(qw) is not None and ensure_eager_tensor(qs) is not None and (qb is not None)
        assert ensure_callable(quantized_matmul)(t, q_conf) is not None
        assert ensure_callable(quantized_matmul)(t, q_conf_t) is not None
        assert ensure_callable(gather_qmm)(t, q_conf) is not None
        assert ensure_callable(gather_qmm)(t, q_conf_t) is not None
        assert ensure_callable(dequantize)(t, t, t) is not None
        assert ensure_callable(quantized_conv)(t, q_conf) is not None
    finally:
        ensure_mutable(config).eager_mode = ensure_eager_tensor(original_eager)
        global_tracing_state.is_tracing = False
    _state.update(ensure_tensor_dict({"original_eager": original_eager, "q_conf": q_conf, "q_conf_t": q_conf_t}))
    _state.update(ensure_tensor_dict({"t": t, "t_max": t_max, "t_min": t_min}))


def _test_quantized_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_quantized_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class DummyShape:
        """Test class DummyShape."""

        shape = (1, 2)

    assert ensure_callable(QuantizeOp)().infer_shape(ensure_callable(DummyShape)()) == (1, 2)
    assert ensure_callable(QuantizedMatmulOp)().infer_shape(ensure_callable(DummyShape)()) == (1, 2)
    assert ensure_callable(GatherQMMOp)().infer_shape(ensure_callable(DummyShape)()) == (1, 2)
    assert ensure_callable(FakeQuantWithMinMaxVarsOp)().infer_shape(ensure_callable(DummyShape)()) == (1, 2)
    assert ensure_callable(QuantizeAndDequantizeOp)().infer_shape(ensure_callable(DummyShape)()) == (1, 2)
    assert ensure_callable(AbsMaxQuantizeOp)().infer_shape(ensure_callable(DummyShape)()) == (1, 2)
    assert ensure_callable(ComputeFloat8AmaxHistoryOp)().infer_shape(ensure_callable(DummyShape)()) == (1, 2)
    assert ensure_callable(ComputeFloat8ScaleOp)().infer_shape(ensure_callable(DummyShape)()) == (1, 2)
    _state.update(ensure_tensor_dict({"DummyShape": DummyShape}))


def test_quantized_coverage() -> None:
    """Test function test_quantized_coverage."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_quantized_coverage_part_1(_state)
    _test_quantized_coverage_part_2(_state)
    _test_quantized_coverage_part_3(_state)
    assert len(_state) > 0


def _test_rnn_coverage_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_rnn_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    config.eager_mode = True
    t_in = Tensor(np.ones((2, 1, 4)), TensorConfig(shape=(2, 1, 4), dtype=DType("float32"), device=Device("cpu")))
    t_h = Tensor(np.ones((1, 4)), TensorConfig(shape=(1, 4), dtype=DType("float32"), device=Device("cpu")))
    w = Tensor(np.ones((4, 4)), TensorConfig(shape=(4, 4), dtype=DType("float32"), device=Device("cpu")))
    rw = Tensor(np.ones((4, 4)), TensorConfig(shape=(4, 4), dtype=DType("float32"), device=Device("cpu")))
    b = Tensor(np.ones((4,)), TensorConfig(shape=(4,), dtype=DType("float32"), device=Device("cpu")))
    assert simple_rnn_cell(t_in[0], (t_h,), w, rw, b) is not None

    def dummy_cell(inputs: EagerValue, state: EagerValue) -> EagerValue:
        """Test function dummy_cell.

        Args:
            inputs: Test parameter `inputs`.
            state: Test parameter `state`.

        Returns:
            EagerValue: Computed output value from `dummy_cell`.
        """
        return ensure_callable(simple_rnn_cell)(inputs, state, w, rw, b)

    (out, final_state) = rnn(t_in, (t_h,), dummy_cell, RNNConfig(time_major=True))
    assert ensure_eager_tensor(out) is not None and final_state is not None
    t_in_batch_major = Tensor(np.ones((1, 2, 4)), TensorConfig(shape=(1, 2, 4), dtype=DType("float32"), device=Device("cpu")))
    (out, final_state) = rnn(t_in_batch_major, (t_h,), dummy_cell, RNNConfig(time_major=False))
    assert ensure_eager_tensor(out) is not None and final_state is not None
    bidi_in = BidirectionalInputs(forward_inputs=t_in, backward_inputs=t_in, forward_initial_state=(t_h,), backward_initial_state=(t_h,))
    (bidi_out, fwd, bwd) = bidirectional(bidi_in, dummy_cell, BidirectionalConfig(merge_mode="concat", time_major=True))
    assert bidi_out is not None
    (bidi_out2, fwd, bwd) = bidirectional(bidi_in, dummy_cell, BidirectionalConfig(merge_mode="sum", time_major=True))
    assert bidi_out2 is not None
    _state.update(ensure_tensor_dict({"b": b, "bidi_in": bidi_in, "bwd": bwd, "dummy_cell": dummy_cell}))
    _state.update(ensure_tensor_dict({"final_state": final_state, "fwd": fwd, "out": bidi_out2, "rw": rw}))
    _state.update(ensure_tensor_dict({"t_h": t_h, "t_in": t_in, "t_in_batch_major": t_in_batch_major, "w": w}))


def _test_rnn_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_rnn_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (bidi_in, dummy_cell, t_h, t_in) = (_state["bidi_in"], _state["dummy_cell"], _state["t_h"], _state["t_in"])
    (out, fwd, bwd) = ensure_callable(bidirectional)(bidi_in, dummy_cell, ensure_callable(BidirectionalConfig)(merge_mode="mul", time_major=True))
    assert out is not None
    (out, fwd, bwd) = ensure_callable(bidirectional)(bidi_in, dummy_cell, ensure_callable(BidirectionalConfig)(merge_mode="ave", time_major=True))
    assert out is not None
    (out, fwd, bwd) = ensure_callable(bidirectional)(bidi_in, dummy_cell, ensure_callable(BidirectionalConfig)(merge_mode="none", time_major=True))
    assert out is not None

    def scan_dummy(carry: EagerValue, x: EagerValue) -> EagerValue:
        """Test function scan_dummy.

        Args:
            carry: Test parameter `carry`.
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `scan_dummy`.
        """
        return dummy_cell(x, carry)

    (carry, ys) = ensure_callable(scan)(scan_dummy, (t_h,), t_in, config=ensure_callable(ScanConfig)(unroll=True, reverse=True))
    assert ensure_eager_tensor(carry) is not None and ys is not None
    original_eager = config.eager_mode
    _state.update(ensure_tensor_dict({"bwd": bwd, "carry": carry, "fwd": fwd, "original_eager": original_eager}))
    _state.update(ensure_tensor_dict({"out": out, "scan_dummy": scan_dummy, "ys": ys}))
    _state.update(ensure_tensor_dict({"bidi_in": bidi_in, "dummy_cell": dummy_cell, "t_h": t_h}))
    _state.update(ensure_tensor_dict({"t_in": t_in}))


def _test_rnn_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_rnn_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (dummy_cell, original_eager, scan_dummy, t_h) = (_state["dummy_cell"], _state["original_eager"], _state["scan_dummy"], _state["t_h"])
    t_in = _state["t_in"]
    try:
        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

        class DummyGraph:
            """Test class DummyGraph."""

            name = "dummy"
            nodes = ensure_tensor_dict({})

            def add_node(self, node: EagerValue) -> None:
                """Test function add_node.

                Args:
                    node: Test parameter `node`.
                """
                pass

        ensure_mutable(global_tracing_state).active_graph = ensure_eager_tensor(DummyGraph())
        with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.cf_scan") as mock_cf_scan:
            mock_cf_scan.return_value = ("carry", "y")
            (carry, ys) = ensure_callable(scan)(scan_dummy, (t_h,), t_in, config=ensure_callable(ScanConfig)(unroll=False, reverse=True))
            assert carry == "carry" and ys == "y"
            (carry, ys) = ensure_callable(scan)(scan_dummy, (t_h,), t_in, config=ensure_callable(ScanConfig)(unroll=False, reverse=False))
            assert carry == "carry" and ys == "y"
    finally:
        ensure_mutable(config).eager_mode = ensure_eager_tensor(original_eager)
        global_tracing_state.is_tracing = False
    dev_wrapper = RNNCellDeviceWrapper(dummy_cell, Device("cpu"))
    assert dev_wrapper(t_in[0], (t_h,)) is not None
    _state.update(ensure_tensor_dict({"dev_wrapper": dev_wrapper}))
    _state.update(ensure_tensor_dict({"dummy_cell": dummy_cell, "original_eager": original_eager, "scan_dummy": scan_dummy}))
    _state.update(ensure_tensor_dict({"t_h": t_h, "t_in": t_in}))


def _test_rnn_coverage_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_rnn_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (dummy_cell, t_h, t_in) = (_state["dummy_cell"], _state["t_h"], _state["t_in"])
    do_wrapper = RNNCellDropoutWrapper(dummy_cell, DropoutWrapperConfig(input_keep_prob=0.5, output_keep_prob=0.5))
    with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.dropout") as mock_do:
        mock_do.return_value = t_in[0]
        assert ensure_callable(do_wrapper)(t_in[0], (t_h,)) is not None
    res_wrapper = RNNCellResidualWrapper(dummy_cell)
    assert res_wrapper(t_in[0], (t_h,)) is not None
    res_wrapper2 = RNNCellResidualWrapper(dummy_cell, residual_fn=lambda x, y: y)
    assert res_wrapper2(t_in[0], (t_h,)) is not None
    _state.update(ensure_tensor_dict({"do_wrapper": do_wrapper, "mock_do": mock_do, "res_wrapper": res_wrapper, "res_wrapper2": res_wrapper2}))


def test_rnn_coverage() -> None:
    """Test function test_rnn_coverage."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_rnn_coverage_part_1(_state)
    _test_rnn_coverage_part_2(_state)
    _test_rnn_coverage_part_3(_state)
    _test_rnn_coverage_part_4(_state)
    assert len(_state) > 0


def test_time_distributed_coverage() -> None:
    """Test function test_time_distributed_coverage."""
    config.eager_mode = True
    t = Tensor(np.ones((2, 2)), TensorConfig(shape=(2, 2), dtype=DType("float32"), device=Device("cpu")))
    assert TimeDistributed().infer_shape(t) == (2, 2)
    with patch("ml_switcheroo_compiler.ops.nn.time_distributed.get_op") as mock_get_op:
        mock_op = mock_get_op.return_value.return_value
        mock_op.return_value = "timed"
        with pytest.raises(Exception, match=".*"):
            ensure_callable(time_distributed)(t, "Dense")


def test_time_distributed_inner_op_shape_inference(mocker: EagerTensorProtocol) -> None:
    """Verify that TimeDistributed unrolls batch and time dimensions across inner ops.

    Args:
        mocker (EagerTensorProtocol): Pytest mocker fixture.
    """
    t = Tensor(None, TensorConfig(shape=(4, 10, 32), dtype=DType("float32"), device=Device("cpu")))

    class MockInnerOp:
        """Test class MockInnerOp."""

        def infer_shape(self, inp: EagerValue, **kwargs: EagerValue) -> tuple[int, ...]:
            """Test function infer_shape.

            Args:
                inp: Test parameter `inp`.
                kwargs: Test parameter `kwargs`.

            Returns:
                tuple[int, ...]: Computed output value from `infer_shape`.
            """
            return ensure_shape((ensure_eager_tensor(inp).shape[0], 64))

    ensure_callable(mocker.patch)("ml_switcheroo_compiler.ops.nn.time_distributed.get_op", return_value=lambda: ensure_callable(MockInnerOp)())
    inferred = TimeDistributed().infer_shape(t, wrapped_op_name="Dense")
    assert inferred == (4, 10, 64)
