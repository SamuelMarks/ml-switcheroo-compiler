"""Unit tests for tracing/tracer.py ensuring 100% line, branch, and function coverage."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ir.core import IRNode
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import (
    ProxyTensor,
    TracerTape,
    _tracer,
    get_trace_count,
    increment_trace_count,
    reset_trace_count,
)


def test_proxy_tensor_binary_op() -> None:
    """Test the correctness and edge cases of the proxy tensor binary op functionality."""
    global_tracing_state.start_tracing()
    p1 = ProxyTensor(id="p1", shape=(2,), dtype="float32")
    p2 = ProxyTensor(id="p2", shape=(2,), dtype="float32")
    out = p1._binary_op(p2, "Add")
    assert out.shape == (2,)
    out_scalar = p1._binary_op(1.0, "Add")
    assert out_scalar.shape == (2,)
    out_unary = p1._unary_op("Neg")
    assert out_unary.shape == (2,)
    global_tracing_state.stop_tracing()


def test_tracing_inputs_and_specs() -> None:
    """Verify active_graph.inputs and input_specs extraction during tracing."""
    g = global_tracing_state.start_tracing("spec_test")
    p1 = ProxyTensor(id="in1", shape=(4, 8), dtype="float32")
    p2 = ProxyTensor(id="in2", shape=(4, 8), dtype="float32")

    assert "in1" in g.inputs
    assert "in2" in g.inputs
    assert g.input_specs["in1"].shape == (4, 8)
    assert g.input_specs["in1"].dtype == "float32"

    out = p1._binary_op(p2, "Add")
    # Output node should not be in inputs
    assert out.id not in g.inputs
    global_tracing_state.stop_tracing()


def test_trace_counts() -> None:
    """Verify trace counter operations including get, increment, and reset."""

    def dummy_fn(x: Tensor) -> Tensor:
        return x

    assert get_trace_count(dummy_fn) == 0
    increment_trace_count(dummy_fn)
    assert get_trace_count(dummy_fn) == 1
    increment_trace_count(dummy_fn)
    assert get_trace_count(dummy_fn) == 2
    reset_trace_count(dummy_fn)
    assert get_trace_count(dummy_fn) == 0
    # Reset when not present does nothing
    reset_trace_count(dummy_fn)
    assert get_trace_count(dummy_fn) == 0


def test_tracer_tape_and_global_instance() -> None:
    """Verify TracerTape life cycle and global _tracer instance methods."""
    tape = TracerTape()
    g = tape.start_tracing("TapeModel")
    assert g is not None
    assert global_tracing_state.is_tracing is True

    node = IRNode(id="n1", op_type="Add", inputs=["a", "b"], outputs=["c"])
    tape.add_node(node)
    assert "n1" in g.nodes

    final_g = tape.stop_tracing()
    assert final_g is not None
    assert global_tracing_state.is_tracing is False

    # Also exercise global _tracer
    g2 = _tracer.start_tracing("GlobalTapeModel")
    assert g2 is not None
    _tracer.add_node(IRNode(id="n2", op_type="Relu", inputs=["c"], outputs=["d"]))
    final_g2 = _tracer.stop_tracing()
    assert final_g2 is not None


def test_proxy_tensor_sparsity_and_branches() -> None:
    """Verify ProxyTensor initialization under various graph states."""
    # 1. Outside tracing
    global_tracing_state.stop_tracing()
    p_outside = ProxyTensor(id="outside", shape=(3, 3), dtype="float32", sparsity={"format": "csr"})
    assert p_outside.sparsity == {"format": "csr"}

    # 2. Inside tracing when active_graph has no input_specs
    g = global_tracing_state.start_tracing("no_specs")
    # Remove input_specs attribute to hit branch
    delattr(g, "input_specs")
    p_no_specs = ProxyTensor(id="p_no_specs", shape=(2, 2))
    assert "p_no_specs" in g.inputs

    # 3. Inside tracing when id is already in inputs
    p_dup_in = ProxyTensor(id="p_no_specs", shape=(2, 2))
    assert g.inputs.count("p_no_specs") == 1

    # 4. Inside tracing when id is already in nodes
    g.nodes["node_present"] = LogicalNode(id="node_present", op_type="Constant")
    p_in_nodes = ProxyTensor(id="node_present", shape=(1,))
    assert "node_present" not in g.inputs

    # 5. When active_graph is None or doesn't have inputs/nodes
    class DummyGraph:
        pass

    global_tracing_state.active_graph = DummyGraph()
    p_dummy = ProxyTensor(id="dummy", shape=(1,))
    assert p_dummy.id == "dummy"

    # Clean up
    global_tracing_state.stop_tracing()
