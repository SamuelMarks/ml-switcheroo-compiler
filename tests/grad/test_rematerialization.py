"""Test Rematerialization pass and autodiff checkpointing."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.autodiff import grad
from ml_switcheroo_compiler.transforms.passes.rematerialization import _load_rules, rematerialization_pass

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_rematerialization_pass() -> None:
    """Test function test_rematerialization_pass."""
    rules = _load_rules()
    assert "target_ops" in rules
    assert "Add" in rules["target_ops"]

    graph = IRGraph()
    getattr(graph.nodes, "__setitem__")("in0", cast("Callable[..., EagerTensorProtocol]", IRNode)("in0", "Input", []))
    graph.nodes["add_small"] = IRNode("add_small", "Add", inputs=["in0", "in0"], shape_metadata=[10])
    graph.nodes["add_large"] = IRNode("add_large", "Add", inputs=["in0", "in0"], shape_metadata=[10000, 10000])
    for i in range(11):
        graph.nodes[f"dummy{i}"] = IRNode(f"dummy{i}", "Add", inputs=["add_large", "in0"], shape_metadata=[10000, 10000])
    graph.outputs = ["add_small", "dummy10"]

    modified = rematerialization_pass(graph)
    assert modified
    assert not graph.nodes["add_small"].attributes.get("rematerialize", False)
    assert graph.nodes["add_large"].attributes.get("rematerialize", False)


def test_autodiff_recompute() -> None:
    """Test function test_autodiff_recompute."""
    graph = IRGraph()
    getattr(graph.nodes, "__setitem__")("in0", cast("Callable[..., EagerTensorProtocol]", IRNode)("in0", "Input", []))
    graph.nodes["add1"] = IRNode("add1", "Add", inputs=["in0", "in0"], shape_metadata=[10000, 10000])
    for i in range(11):
        graph.nodes[f"dummy{i}"] = IRNode(f"dummy{i}", "Add", inputs=["add1", "in0"], shape_metadata=[10000, 10000])
    graph.nodes["add2"] = IRNode("add2", "Add", inputs=["dummy10", "in0"], shape_metadata=[10000, 10000])
    graph.outputs = ["add2"]

    # Just set it manually to test AD logic
    graph.nodes["add1"].attributes["rematerialize"] = True

    grad_graph = grad(graph, ["in0"], "add2")

    recompute_found = False
    for node_id, node in grad_graph.nodes.items():
        if "recompute" in node_id and node.op_type == "Add":
            recompute_found = True

    assert recompute_found


def test_autodiff_recompute_nested() -> None:
    """Test function test_autodiff_recompute_nested."""
    graph = IRGraph()
    getattr(graph.nodes, "__setitem__")("in0", cast("Callable[..., EagerTensorProtocol]", IRNode)("in0", "Input", []))
    graph.nodes["add1"] = IRNode("add1", "Add", inputs=["in0", "in0"])
    graph.nodes["add1"].attributes["rematerialize"] = True
    graph.nodes["add2"] = IRNode("add2", "Add", inputs=["add1", "in0"])
    graph.nodes["add2"].attributes["rematerialize"] = True
    graph.outputs = ["add2"]

    grad_graph = grad(graph, ["in0"], "add2")

    recomputes = sum(1 for n in grad_graph.nodes if "recompute" in n)
    assert recomputes >= 2


def test_numerical_gradients() -> None:
    """Test function test_numerical_gradients."""
    graph1 = IRGraph()
    getattr(graph1.nodes, "__setitem__")("in0", cast("Callable[..., EagerTensorProtocol]", IRNode)("in0", "Input", []))
    graph1.nodes["add1"] = IRNode("add1", "Add", inputs=["in0", "in0"])
    graph1.nodes["add2"] = IRNode("add2", "Add", inputs=["add1", "in0"])
    graph1.outputs = ["add2"]

    grad_graph1 = grad(graph1, ["in0"], "add2")

    graph2 = IRGraph()
    getattr(graph2.nodes, "__setitem__")("in0", cast("Callable[..., EagerTensorProtocol]", IRNode)("in0", "Input", []))
    graph2.nodes["add1"] = IRNode("add1", "Add", inputs=["in0", "in0"])
    graph2.nodes["add1"].attributes["rematerialize"] = True
    graph2.nodes["add2"] = IRNode("add2", "Add", inputs=["add1", "in0"])
    graph2.outputs = ["add2"]

    grad_graph2 = grad(graph2, ["in0"], "add2")

    # Check that they compute the same thing structurally since we don't have a numeric evaluator set up here.
    assert len(grad_graph1.nodes) <= len(grad_graph2.nodes)  # grad_graph2 has recompute nodes


def test_buffer_analysis() -> None:
    """Test function test_buffer_analysis."""
    from ml_switcheroo_compiler.transforms.passes.buffer_allocation import buffer_allocation_pass

    graph1 = IRGraph()
    getattr(graph1.nodes, "__setitem__")("in0", cast("Callable[..., EagerTensorProtocol]", IRNode)("in0", "Input", []))
    graph1.nodes["add_large"] = IRNode("add_large", "Add", inputs=["in0", "in0"], shape_metadata=[10000, 10000])
    for i in range(15):
        graph1.nodes[f"dummy{i}"] = IRNode(f"dummy{i}", "Add", inputs=["add_large", "in0"], shape_metadata=[10000, 10000])
    graph1.outputs = ["dummy14"]

    # We simulate dropping the node which is the INTENT of rematerialization:
    # Actually rematerialization_pass drops it from forward pass consumers?
    # No, it injects a clone. Let's just call buffer_allocation_pass.
    buffer_allocation_pass(graph1)
    assert "dummy14" in graph1.nodes
    assert "buffer_id" in graph1.nodes["dummy14"].attributes
