"""Tests for test_rnn_quantized."""

from __future__ import annotations

import typing
from unittest.mock import patch

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalNode
from pytest_mock import MockerFixture

from ml_switcheroo_compiler.backends.eager_registry import BackendArg, unpack_triple
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
from ml_switcheroo_compiler.ops.nn.rnn_utils import (
    BidirectionalConfig,
    BidirectionalInputs,
    DropoutWrapperConfig,
    RNNCellDeviceWrapper,
    RNNCellDropoutWrapper,
    RNNCellResidualWrapper,
    RNNConfig,
    ScanConfig,
    bidirectional,
    rnn,
    scan,
)
from ml_switcheroo_compiler.ops.nn.time_distributed import TimeDistributed, time_distributed
from tests.testing_utils import (
    EagerBoundCallable,
    MockTensorStub,
    coerce_operand_list,
    coerce_shape,
    ensure_tensor,
    extract_test_shape,
    require_callable,
    unpack_pair,
)


class _TestQuantizedCoverageState(typing.TypedDict, total=False):
    """Typed shared state for `_test_quantized_coverage` split helpers."""

    original_eager: bool
    q_conf: QuantizedOpsConfig
    q_conf_t: QuantizedOpsConfig
    t: Tensor
    t_max: Tensor
    t_min: Tensor


def _test_quantized_coverage_part_1(_state: _TestQuantizedCoverageState) -> None:
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
    (q, s) = unpack_pair(abs_max_quantize(t))
    assert q is not None and s is not None
    assert compute_float8_amax_history(t, t) is not None
    assert compute_float8_scale(t, t) is not None
    (qw, qs, qb) = unpack_triple(quantize(t))
    assert qw is not None and qs is not None and (qb is not None)
    q_conf = QuantizedOpsConfig(weight=t, scales=t, biases=t, indices=t, q_config=QuantizationConfig(transpose=False))
    q_conf_t = QuantizedOpsConfig(weight=t, scales=t, biases=t, indices=t, q_config=QuantizationConfig(transpose=True))
    assert quantized_matmul(t, q_conf) is not None
    assert gather_qmm(t, q_conf) is not None
    assert quantized_linear(t, q_conf) is not None
    t_idx = Tensor(np.array([[0, 0]]), TensorConfig(shape=(1, 2), dtype=DType("int32"), device=Device("cpu")))
    assert quantized_embedding(t_idx, q_conf) is not None
    original_eager = config.eager_mode
    _state.update({"original_eager": original_eager, "q_conf": q_conf, "q_conf_t": q_conf_t})
    _state.update({"t": t, "t_max": t_max, "t_min": t_min})


def _test_quantized_coverage_part_2(_state: _TestQuantizedCoverageState) -> None:
    """Execute part 2 of test_quantized_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (original_eager, q_conf, q_conf_t, t) = (
        _state["original_eager"],
        _state["q_conf"],
        _state["q_conf_t"],
        _state["t"],
    )
    (t_max, t_min) = (_state["t_max"], _state["t_min"])
    try:
        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

        class DummyGraph:
            """Test class DummyGraph."""

            name = "dummy"
            nodes: dict[str, LogicalNode] = {}

            def add_node(self, node: BackendArg) -> None:
                """Test function add_node.

                Args:
                    node: Test parameter `node`.
                """
                pass

        pytest.MonkeyPatch().setattr(global_tracing_state, "active_graph", DummyGraph(), raising=False)
        assert fake_quant_with_min_max_vars(t, t_min, t_max) is not None
        assert quantize_and_dequantize(t, t_min, t_max) is not None
        (q, s) = unpack_pair(abs_max_quantize(t))
        assert q is not None and s is not None
        assert compute_float8_amax_history(t, t) is not None
        assert compute_float8_scale(t, t) is not None
        (qw, qs, qb) = unpack_triple(quantize(t))
        assert qw is not None and qs is not None and (qb is not None)
        assert quantized_matmul(t, q_conf) is not None
        assert quantized_matmul(t, q_conf_t) is not None
        assert gather_qmm(t, q_conf) is not None
        assert gather_qmm(t, q_conf_t) is not None
        assert dequantize(t, t, t) is not None
        assert quantized_conv(t, q_conf) is not None
    finally:
        pytest.MonkeyPatch().setattr(config, "eager_mode", original_eager, raising=False)
        global_tracing_state.is_tracing = False
    _state.update({"original_eager": original_eager, "q_conf": q_conf, "q_conf_t": q_conf_t})
    _state.update({"t": t, "t_max": t_max, "t_min": t_min})


def _test_quantized_coverage_part_3(_state: _TestQuantizedCoverageState) -> None:
    """Execute part 3 of test_quantized_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class DummyShape(MockTensorStub):
        """Test class DummyShape."""

        def __init__(self) -> None:
            """Initialize stub attributes."""
            super().__init__()
            self.shape = (1, 2)

    assert QuantizeOp().infer_shape(DummyShape()) == (1, 2)
    assert QuantizedMatmulOp().infer_shape(DummyShape()) == (1, 2)
    assert GatherQMMOp().infer_shape(DummyShape()) == (1, 2)
    assert FakeQuantWithMinMaxVarsOp().infer_shape(DummyShape()) == (1, 2)
    assert QuantizeAndDequantizeOp().infer_shape(DummyShape()) == (1, 2)
    assert AbsMaxQuantizeOp().infer_shape(DummyShape()) == (1, 2)
    assert ComputeFloat8AmaxHistoryOp().infer_shape(DummyShape()) == (1, 2)
    assert ComputeFloat8ScaleOp().infer_shape(DummyShape()) == (1, 2)


def test_quantized_coverage() -> None:
    """Test function test_quantized_coverage."""
    _state: _TestQuantizedCoverageState = {}
    _test_quantized_coverage_part_1(_state)
    _test_quantized_coverage_part_2(_state)
    _test_quantized_coverage_part_3(_state)
    assert len(_state) > 0


class _TestRnnCoverageState(typing.TypedDict, total=False):
    """Typed shared state for `_test_rnn_coverage` split helpers."""

    bidi_in: BidirectionalInputs
    dummy_cell: EagerBoundCallable
    original_eager: bool
    scan_dummy: EagerBoundCallable
    t_h: Tensor
    t_in: Tensor


def _test_rnn_coverage_part_1(_state: _TestRnnCoverageState) -> None:
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

    def dummy_cell(inputs: BackendArg, state: BackendArg) -> BackendArg:
        """Test function dummy_cell.

        Args:
            inputs: Test parameter `inputs`.
            state: Test parameter `state`.

        Returns:
            BackendArg: Computed output value from `dummy_cell`.
        """
        state_tup = tuple(ensure_tensor(s) for s in coerce_operand_list(state))
        return simple_rnn_cell(ensure_tensor(inputs), state_tup, w, rw, b)

    (out, final_state) = rnn(t_in, (t_h,), dummy_cell, RNNConfig(time_major=True))
    assert out.shape == (2, 1, 4) and len(final_state) == 1
    t_in_batch_major = Tensor(
        np.ones((1, 2, 4)), TensorConfig(shape=(1, 2, 4), dtype=DType("float32"), device=Device("cpu"))
    )
    (out, final_state) = rnn(t_in_batch_major, (t_h,), dummy_cell, RNNConfig(time_major=False))
    assert out.shape == (1, 2, 4) and len(final_state) == 1
    bidi_in = BidirectionalInputs(
        forward_inputs=t_in, backward_inputs=t_in, forward_initial_state=(t_h,), backward_initial_state=(t_h,)
    )
    (bidi_out, fwd, bwd) = bidirectional(bidi_in, dummy_cell, BidirectionalConfig(merge_mode="concat", time_major=True))
    assert bidi_out is not None
    (bidi_out2, fwd, bwd) = bidirectional(bidi_in, dummy_cell, BidirectionalConfig(merge_mode="sum", time_major=True))
    assert bidi_out2 is not None
    _state.update({"bidi_in": bidi_in, "dummy_cell": require_callable(dummy_cell)})
    _state.update({"t_h": t_h, "t_in": t_in})


def _test_rnn_coverage_part_2(_state: _TestRnnCoverageState) -> None:
    """Execute part 2 of test_rnn_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (bidi_in, dummy_cell, t_h, t_in) = (_state["bidi_in"], _state["dummy_cell"], _state["t_h"], _state["t_in"])
    (out, fwd, bwd) = bidirectional(bidi_in, dummy_cell, BidirectionalConfig(merge_mode="mul", time_major=True))
    assert out is not None
    (out, fwd, bwd) = bidirectional(bidi_in, dummy_cell, BidirectionalConfig(merge_mode="ave", time_major=True))
    assert out is not None
    (out, fwd, bwd) = bidirectional(bidi_in, dummy_cell, BidirectionalConfig(merge_mode="none", time_major=True))
    assert out is not None

    def scan_dummy(carry: BackendArg, x: BackendArg) -> BackendArg:
        """Test function scan_dummy.

        Args:
            carry: Test parameter `carry`.
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `scan_dummy`.
        """
        return dummy_cell(x, carry)

    (carry, ys) = scan(scan_dummy, (t_h,), t_in, config=ScanConfig(unroll=True, reverse=True))
    assert carry is not None and ys is not None
    original_eager = config.eager_mode
    _state.update({"original_eager": original_eager})
    _state.update({"scan_dummy": require_callable(scan_dummy)})
    _state.update({"bidi_in": bidi_in, "dummy_cell": require_callable(dummy_cell), "t_h": t_h})
    _state.update({"t_in": t_in})


def _test_rnn_coverage_part_3(_state: _TestRnnCoverageState) -> None:
    """Execute part 3 of test_rnn_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (dummy_cell, original_eager, scan_dummy, t_h) = (
        _state["dummy_cell"],
        _state["original_eager"],
        _state["scan_dummy"],
        _state["t_h"],
    )
    t_in = _state["t_in"]
    try:
        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

        class DummyGraph:
            """Test class DummyGraph."""

            name = "dummy"
            nodes: dict[str, LogicalNode] = {}

            def add_node(self, node: BackendArg) -> None:
                """Test function add_node.

                Args:
                    node: Test parameter `node`.
                """
                pass

        pytest.MonkeyPatch().setattr(global_tracing_state, "active_graph", DummyGraph(), raising=False)
        y_sentinel = Tensor(None, TensorConfig((1,), DType.Float32, Device("cpu")))
        with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.cf_scan") as mock_cf_scan:
            mock_cf_scan.return_value = ("carry", y_sentinel)
            (carry, ys) = scan(scan_dummy, (t_h,), t_in, config=ScanConfig(unroll=False, reverse=True))
            assert carry == "carry" and id(ys) == id(y_sentinel)
            (carry, ys) = scan(scan_dummy, (t_h,), t_in, config=ScanConfig(unroll=False, reverse=False))
            assert carry == "carry" and id(ys) == id(y_sentinel)
    finally:
        pytest.MonkeyPatch().setattr(config, "eager_mode", original_eager, raising=False)
        global_tracing_state.is_tracing = False
    dev_wrapper = RNNCellDeviceWrapper(dummy_cell, Device("cpu"))
    assert dev_wrapper(t_in[0], (t_h,)) is not None
    _state.update(
        {
            "dummy_cell": require_callable(dummy_cell),
            "original_eager": original_eager,
            "scan_dummy": require_callable(scan_dummy),
        }
    )
    _state.update({"t_h": t_h, "t_in": t_in})


def _test_rnn_coverage_part_4(_state: _TestRnnCoverageState) -> None:
    """Execute part 4 of test_rnn_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (dummy_cell, t_h, t_in) = (_state["dummy_cell"], _state["t_h"], _state["t_in"])
    do_wrapper = RNNCellDropoutWrapper(dummy_cell, DropoutWrapperConfig(input_keep_prob=0.5, output_keep_prob=0.5))
    with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.dropout") as mock_do:
        mock_do.return_value = t_in[0]
        assert do_wrapper(t_in[0], (t_h,)) is not None
    res_wrapper = RNNCellResidualWrapper(dummy_cell)
    assert res_wrapper(t_in[0], (t_h,)) is not None
    res_wrapper2 = RNNCellResidualWrapper(dummy_cell, residual_fn=lambda x, y: y)
    assert res_wrapper2(t_in[0], (t_h,)) is not None


def test_rnn_coverage() -> None:
    """Test function test_rnn_coverage."""
    _state: _TestRnnCoverageState = {}
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
            time_distributed(t, "Dense")


def test_time_distributed_inner_op_shape_inference(mocker: MockerFixture) -> None:
    """Verify that TimeDistributed unrolls batch and time dimensions across inner ops.

    Args:
        mocker (MockerFixture): Pytest mocker fixture.
    """
    t = Tensor(None, TensorConfig(shape=(4, 10, 32), dtype=DType("float32"), device=Device("cpu")))

    class MockInnerOp:
        """Test class MockInnerOp."""

        def infer_shape(self, inp: BackendArg, **kwargs: BackendArg) -> tuple[int, ...]:
            """Test function infer_shape.

            Args:
                inp: Test parameter `inp`.
                kwargs: Test parameter `kwargs`.

            Returns:
                tuple[int, ...]: Computed output value from `infer_shape`.
            """
            return coerce_shape((extract_test_shape(inp)[0], 64))

    mocker.patch("ml_switcheroo_compiler.ops.nn.time_distributed.get_op", return_value=lambda: MockInnerOp())
    inferred = TimeDistributed().infer_shape(t, wrapped_op_name="Dense")
    assert inferred == (4, 10, 64)
