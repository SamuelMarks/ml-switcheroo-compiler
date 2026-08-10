"""Unit tests for Dead Code Elimination (DCE) pass."""

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.dce import _find_side_effect_nodes, dce_pass


def test_dce_basic():
    """Test DCE on a simple graph with dead nodes."""
    graph = IRGraph()
    graph.nodes = {
        "n1": IRNode(id="n1", op_type="Input", inputs=[]),
        "n2": IRNode(id="n2", op_type="Add", inputs=["n1", "n1"]),
        "n3": IRNode(id="n3", op_type="Mul", inputs=["n2", "n1"]),  # dead
        "n4": IRNode(id="n4", op_type="Sub", inputs=["n2", "n2"]),
    }
    graph.outputs = ["n4"]

    assert dce_pass(graph) is True
    assert "n3" not in graph.nodes
    assert "n4" in graph.nodes
    assert "n2" in graph.nodes
    assert "n1" in graph.nodes


def test_dce_chained_dead_nodes():
    """Test DCE on a chain of dead nodes."""
    graph = IRGraph()
    graph.nodes = {
        "n1": IRNode(id="n1", op_type="Input", inputs=[]),
        "n2": IRNode(id="n2", op_type="Add", inputs=["n1", "n1"]),  # dead
        "n3": IRNode(id="n3", op_type="Add", inputs=["n2", "n2"]),  # dead
    }
    graph.outputs = ["n1"]

    assert dce_pass(graph) is True
    assert "n2" not in graph.nodes
    assert "n3" not in graph.nodes


def test_dce_side_effect_preservation():
    """Test that DCE preserves nodes with side effects."""
    graph = IRGraph()
    graph.nodes = {
        "n1": IRNode(id="n1", op_type="Input", inputs=[]),
        "n2": IRNode(id="n2", op_type="Seed", inputs=["n1"]),  # Side effect, preserve
        "n3": IRNode(id="n3", op_type="Add", inputs=["n1", "n1"]),  # dead
    }
    graph.outputs = ["n1"]

    assert _find_side_effect_nodes(graph) == {"n2"}
    assert dce_pass(graph) is True
    assert "n3" not in graph.nodes
    assert "n2" in graph.nodes  # Preserved
    assert "n1" in graph.nodes


def test_dce_no_op():
    """Test DCE on a graph where all nodes are reachable."""
    graph = IRGraph()
    graph.nodes = {"n1": IRNode(id="n1", op_type="Input", inputs=[]), "n2": IRNode(id="n2", op_type="Add", inputs=["n1", "n1"])}
    graph.outputs = ["n2"]

    assert dce_pass(graph) is False
    assert len(graph.nodes) == 2
