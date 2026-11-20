"""Unit tests for the tracer tape and proxy tensor components of the ML Switcheroo tracing.

system

This module verifies the behavior of tracing contexts, mathematical operations on proxy
tensors, error handling outside active tracing contexts, and AST reference
propagation.
"""

from __future__ import annotations

import types
import typing

import pytest
from ml_switcheroo_ir import IRNode, LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import ShapeMismatchError, TracingError
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.state import create_read_variable
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.tracing.state import TracingState, global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import (
    ProxyTensor,
    TracerTape,
    get_trace_count,
    increment_trace_count,
    reset_trace_count,
)
from tests.testing_utils import adapt_backend_module, coerce_operand_list

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg
    from ml_switcheroo_compiler.core.config import Config
    from ml_switcheroo_compiler.tracing.autograph import LoopOptions


def testglobal_tracing_state_tape() -> None:
    """Verifies the lifecycle and state transitions of the TracerTape."""
    tape = TracerTape()

    def is_tracing_fn() -> bool:
        """Return active tracing flag.

        Returns:
            bool: Active tracing flag.
        """
        return bool(global_tracing_state.is_tracing)

    assert not is_tracing_fn()
    tape.add_node(LogicalNode(id="n", op_type="Linear"))
    graph = tape.start_tracing("Test")
    assert is_tracing_fn()
    assert graph.name == "Test"
    n = LogicalNode(id="n1", op_type="Input")
    tape.add_node(n)
    assert global_tracing_state.active_graph is not None
    assert "n1" in global_tracing_state.active_graph.nodes
    out_graph = tape.stop_tracing()
    assert not is_tracing_fn()
    assert out_graph is not None
    assert out_graph.nodes["n1"] == n


def test_proxy_tensor_math() -> None:
    """Test the proxy tensor math behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    global_tracing_state.start_tracing()
    a = ProxyTensor(id="a", shape=(2, 3))
    b = ProxyTensor(id="b", shape=(2, 3))
    c = a + b
    assert c.shape == (2, 3)
    assert adapt_backend_module(c.data).require_symbol("id") != "a"
    _ = a - b
    _ = a * b
    _ = a / b
    _ = a**2
    _ = 2 + a
    _ = 3 - a
    _ = 4 * a
    _ = 5 / a
    with pytest.raises((ValueError, ShapeMismatchError)):
        _ = a @ 2
    m = ProxyTensor(id="m", shape=(3, 4))
    n = a @ m
    assert n.shape == (2, 4)
    graph = global_tracing_state.stop_tracing()
    assert len(adapt_backend_module(graph).require_symbol("nodes")) > 0


def test_proxy_tensor_outside_context() -> None:
    """Test the proxy tensor outside context behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    config.eager_mode = False
    a = ProxyTensor(id="a", shape=(2, 3))
    b = ProxyTensor(id="b", shape=(2, 3))
    with pytest.raises((RuntimeError, TracingError)):
        _ = a + b
    with pytest.raises((RuntimeError, TracingError)):
        _ = a @ b
    config.eager_mode = True


def test_tracer_add_node_with_ast_ref() -> None:
    """Test the tracer add node with ast ref behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    tape = TracerTape()
    tape.start_tracing("Test")
    n = LogicalNode(id="n1", op_type="Input", source_ast_ref="test:1")
    tape.add_node(n)
    out_graph = tape.stop_tracing()
    assert adapt_backend_module(out_graph).require_symbol("nodes")["n1"].source_ast_ref == "test:1"


def test_proxy_tensor_assign_operations() -> None:
    """Test the proxy tensor assign operations behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    graph = global_tracing_state.start_tracing(name="assign_test")
    try:
        var_node = create_read_variable("my_var", shape=(2, 2), dtype="float32")
        global_tracing_state.add_node(var_node)
        var_proxy = ProxyTensor(id=var_node.id, shape=(2, 2), dtype="float32")
        val_proxy = ProxyTensor(id="val_1", shape=(2, 2), dtype="float32")
        updated_1 = var_proxy.assign(val_proxy)
        assert updated_1.shape == (2, 2)
        assign_node_1 = graph.nodes[updated_1.id]
        assert assign_node_1.op_type == "AssignVariable"
        assert assign_node_1.attributes["variable_name"] == "my_var"
        updated_2 = updated_1.assign_add(val_proxy)
        assign_node_2 = graph.nodes[updated_2.id]
        assert assign_node_2.op_type == "AssignVariable"
        assert assign_node_2.attributes["variable_name"] == "my_var"
        updated_3 = updated_2.assign_sub(val_proxy)
        assign_node_3 = graph.nodes[updated_3.id]
        assert assign_node_3.op_type == "AssignVariable"
        assert assign_node_3.attributes["variable_name"] == "my_var"
    finally:
        global_tracing_state.stop_tracing()


def test_proxy_tensor_assign_errors() -> None:
    """Test the proxy tensor assign errors behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    var_proxy = ProxyTensor(id="foo", shape=(), dtype="float32")
    val_proxy = ProxyTensor(id="bar", shape=(), dtype="float32")
    with pytest.raises((RuntimeError, TracingError)):
        var_proxy.assign(val_proxy)
    global_tracing_state.start_tracing(name="assign_err_test")
    try:
        with pytest.raises(
            (ValueError, ShapeMismatchError), match="assign\\(\\) can only be called on a variable proxy."
        ):
            var_proxy.assign(val_proxy)
        global_tracing_state.add_node(IRNode(id="foo", op_type="Add", inputs=[], shape_metadata=()))
        with pytest.raises(
            (ValueError, ShapeMismatchError), match="assign\\(\\) can only be called on a variable proxy."
        ):
            var_proxy.assign(val_proxy)
        global_tracing_state.add_node(
            IRNode(
                id="var_node", op_type="ReadVariable", inputs=[], attributes={"variable_name": "x"}, shape_metadata=()
            )
        )
        var_proxy_2 = ProxyTensor(id="var_node", shape=(), dtype="float32")
        out = var_proxy_2.assign(42.0)
        assert out.shape == ()
        assert (
            adapt_backend_module(global_tracing_state.active_graph).require_symbol("nodes")[out.id].op_type
            == "AssignVariable"
        )
    finally:
        global_tracing_state.stop_tracing()


def test_trace_counts() -> None:
    """Test the trace counts behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """

    def my_func() -> None:
        """Evaluate and process the my func operation.

        Returns:
            OpValue: The evaluated or processed output.
        """
        pass

    reset_trace_count(my_func)
    assert get_trace_count(my_func) == 0
    increment_trace_count(my_func)
    assert get_trace_count(my_func) == 1
    reset_trace_count(my_func)
    assert get_trace_count(my_func) == 0

    def traced_fn() -> BackendArg:
        """Evaluate and process the traced fn operation.

        Returns:
            BackendArg: The evaluated or processed output.
        """
        proxy = ProxyTensor(id="out", shape=(), dtype="float32")
        return Tensor(proxy, TensorConfig((), DType.Float32, "cpu"))

    reset_trace_count(traced_fn)
    _trace_function(traced_fn, (), "test_trace")
    assert get_trace_count(traced_fn) == 1
    reset_trace_count(traced_fn)


class _TestTracingStateFullBranchCoverageState(typing.TypedDict, total=False):
    """Typed shared state for `_test_tracing_state_full_branch_coverage` split helpers."""

    LogicalNode: type[LogicalNode]
    SimpleNamespace: type
    TracingState: type[TracingState]
    config: Config
    g1: LogicalGraph
    g2: LogicalGraph
    loop_node: LogicalNode
    old_stream: str | None
    opts: LoopOptions
    state: TracingState
    sys: types.ModuleType


def _test_tracing_state_full_branch_coverage_part_1(_state: _TestTracingStateFullBranchCoverageState) -> None:
    """Execute part 1 of test_tracing_state_full_branch_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    import sys
    from types import SimpleNamespace

    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.tracing.autograph import LoopOptions
    from ml_switcheroo_compiler.tracing.state import TracingState

    state = TracingState()
    state.add_node(LogicalNode(id="n0", op_type="Linear"))

    def active_g() -> LogicalGraph | None:
        """Return active LogicalGraph on `state`.

        Returns:
            LogicalGraph | None: Active graph.
        """
        return state.active_graph

    assert active_g() is None
    g1 = state.start_tracing("Outer")
    assert state.is_tracing
    ag1 = active_g()
    assert ag1 is not None and ag1.name == "Outer"
    opts = LoopOptions(parallel_iterations=4, swap_memory=True, maximum_iterations=100, shape_invariants=[])
    pytest.MonkeyPatch().setattr(state, "current_loop_options", opts, raising=False)
    g2 = state.start_tracing("Inner")
    ag2 = active_g()
    assert ag2 is not None and ag2.name == "Inner"
    assert len(state.graph_stack) == 1
    assert len(state.loop_options_stack) == 1
    node_with_ast = LogicalNode(id="n_ast", op_type="Relu")
    node_with_ast.source_ast_ref = "custom_ref"
    node_with_ast.domain = "custom_domain"
    state.add_node(node_with_ast)
    assert node_with_ast.source_ast_ref == "custom_ref"
    assert node_with_ast.domain == "custom_domain"
    _state.update(
        {"LogicalNode": LogicalNode, "SimpleNamespace": SimpleNamespace, "TracingState": TracingState, "g1": g1}
    )
    _state.update({"g2": g2, "opts": opts, "state": state})
    _state.update({"sys": sys})


def _test_tracing_state_full_branch_coverage_part_2(_state: _TestTracingStateFullBranchCoverageState) -> None:
    """Execute part 2 of test_tracing_state_full_branch_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (LogicalNode, state, sys) = (_state["LogicalNode"], _state["state"], _state["sys"])
    from ml_switcheroo_compiler.core.config import config as real_config
    from tests.testing_utils import MockGraphStub

    assert state.active_graph is not None
    mock_active = MockGraphStub(nodes=state.active_graph.nodes)
    state.active_graph = mock_active
    _state["g2"] = mock_active
    pytest.MonkeyPatch().setattr(state.active_graph, "name", None, raising=False)
    node_no_domain = LogicalNode(id="n_no_dom", op_type="Relu")
    pytest.MonkeyPatch().setattr(node_no_domain, "domain", "", raising=False)
    state.add_node(node_no_domain)
    assert node_no_domain.domain == ""
    pytest.MonkeyPatch().setattr(state.active_graph, "name", "Inner", raising=False)
    node_empty_domain = LogicalNode(id="n_emp_dom", op_type="Relu")
    pytest.MonkeyPatch().setattr(node_empty_domain, "domain", "", raising=False)
    state.add_node(node_empty_domain)
    assert node_empty_domain.domain == "Inner"
    old_stream = real_config.current_stream
    _state.update({"config": real_config, "old_stream": old_stream})
    _state.update({"LogicalNode": LogicalNode, "state": state, "sys": sys})


def _test_tracing_state_full_branch_coverage_part_3(_state: _TestTracingStateFullBranchCoverageState) -> None:
    """Execute part 3 of test_tracing_state_full_branch_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (LogicalNode, config, old_stream, opts) = (
        _state["LogicalNode"],
        _state["config"],
        _state["old_stream"],
        _state["opts"],
    )
    (state, sys) = (_state["state"], _state["sys"])
    try:
        pytest.MonkeyPatch().setattr(config, "current_stream", "stream_cuda_1", raising=False)
        node_stream = LogicalNode(id="n_stream", op_type="Relu")
        pytest.MonkeyPatch().setattr(node_stream, "stream", None, raising=False)
        state.add_node(node_stream)
        assert node_stream.stream == "stream_cuda_1"
        pytest.MonkeyPatch().setattr(config, "current_stream", "default", raising=False)
        node_stream_def = LogicalNode(id="n_stream_def", op_type="Relu")
        pytest.MonkeyPatch().setattr(node_stream_def, "stream", None, raising=False)
        state.add_node(node_stream_def)
        assert node_stream_def.stream is None
        cfg_module = sys.modules.pop("ml_switcheroo_compiler.core.config")
        try:
            node_dummy = LogicalNode(id="n_dummy", op_type="Relu")
            state._enrich_stream(node_dummy)
        finally:
            sys.modules["ml_switcheroo_compiler.core.config"] = cfg_module
    finally:
        config.current_stream = old_stream or "default"
    state.current_loop_options = opts
    loop_node = LogicalNode(id="loop_1", op_type="Loop", attributes={})
    state.add_node(loop_node)
    assert loop_node.attributes["parallel_iterations"] == 4
    assert bool(loop_node.attributes["swap_memory"]) is True
    _state.update({"loop_node": loop_node})
    _state.update({"LogicalNode": LogicalNode, "config": config, "old_stream": old_stream})
    _state.update({"opts": opts, "state": state, "sys": sys})


def _test_tracing_state_full_branch_coverage_part_4(_state: _TestTracingStateFullBranchCoverageState) -> None:
    """Execute part 4 of test_tracing_state_full_branch_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.tracing.autograph import LoopOptions

    (LogicalNode, SimpleNamespace, g2, loop_node) = (
        _state["LogicalNode"],
        _state["SimpleNamespace"],
        _state["g2"],
        _state["loop_node"],
    )
    (opts, state) = (_state["opts"], _state["state"])
    assert loop_node.attributes["maximum_iterations"] == 100
    assert loop_node.attributes["shape_invariants"] == []
    assert loop_node.attributes["loop_options"] is opts
    opts_none = LoopOptions(parallel_iterations=None, swap_memory=None, maximum_iterations=None, shape_invariants=None)
    state.current_loop_options = opts_none
    loop_node_2 = LogicalNode(id="loop_2", op_type="WhileLoop", attributes={"loop_options": "existing"})
    state.add_node(loop_node_2)
    assert "parallel_iterations" not in loop_node_2.attributes
    assert loop_node_2.attributes["loop_options"] == "existing"
    assert state.active_graph is not None
    pytest.MonkeyPatch().setattr(state.active_graph, "inputs", ["input_to_replace", "other_input"], raising=False)
    pytest.MonkeyPatch().setattr(state.active_graph, "input_specs", {"input_to_replace": "spec"}, raising=False)
    replacing_node = LogicalNode(id="input_to_replace", op_type="Add")
    state.add_node(replacing_node)
    assert "input_to_replace" not in coerce_operand_list(
        adapt_backend_module(state.active_graph).require_symbol("inputs")
    )
    specs_after = adapt_backend_module(state.active_graph).get_symbol("input_specs", {})
    assert isinstance(specs_after, dict) and "input_to_replace" not in specs_after
    pytest.MonkeyPatch().delattr(state.active_graph, "input_specs")
    pytest.MonkeyPatch().setattr(state.active_graph, "inputs", ["input_to_replace_2"], raising=False)
    replacing_node_2 = LogicalNode(id="input_to_replace_2", op_type="Mul")
    state.add_node(replacing_node_2)
    assert "input_to_replace_2" not in coerce_operand_list(
        adapt_backend_module(state.active_graph).require_symbol("inputs")
    )
    inner_g = state.stop_tracing()
    assert inner_g is g2
    _state.update({"LogicalNode": LogicalNode, "SimpleNamespace": SimpleNamespace, "g2": g2})
    _state.update({"loop_node": loop_node, "opts": opts, "state": state})


def _test_tracing_state_full_branch_coverage_part_5(_state: _TestTracingStateFullBranchCoverageState) -> None:
    """Execute part 5 of test_tracing_state_full_branch_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (TracingState, g1, opts, state) = (_state["TracingState"], _state["g1"], _state["opts"], _state["state"])

    def is_tr() -> bool:
        """Return state tracing flag.

        Returns:
            bool: Tracing flag.
        """
        return bool(state.is_tracing)

    def act_g() -> LogicalGraph | None:
        """Return state active graph.

        Returns:
            LogicalGraph | None: Active graph.
        """
        return state.active_graph

    def cur_lo() -> LoopOptions | None:
        """Return state current loop options.

        Returns:
            LoopOptions | None: Current loop options.
        """
        lo = state.current_loop_options
        return lo if isinstance(lo, type(opts)) else None

    assert is_tr()
    assert act_g() is g1
    assert cur_lo() is opts
    outer_g = state.stop_tracing()
    assert outer_g is g1
    assert not is_tr()
    assert act_g() is None
    assert cur_lo() is None
    state_dummy = TracingState()
    state_dummy.start_tracing("Base")
    dummy_inner = state_dummy.start_tracing("Inner")
    state_dummy.loop_options_stack.clear()
    popped = state_dummy.stop_tracing()
    assert popped is dummy_inner
    assert state_dummy.current_loop_options is None


def test_tracing_state_full_branch_coverage() -> None:
    """Verify all branches and conditions in TracingState."""
    _state: _TestTracingStateFullBranchCoverageState = {}
    _test_tracing_state_full_branch_coverage_part_1(_state)
    _test_tracing_state_full_branch_coverage_part_2(_state)
    _test_tracing_state_full_branch_coverage_part_3(_state)
    _test_tracing_state_full_branch_coverage_part_4(_state)
    _test_tracing_state_full_branch_coverage_part_5(_state)
    assert len(_state) > 0
