"""Unit tests for Common Subexpression Elimination (CSE) pass."""

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.cse import cse_pass


def test_cse_basic():
    """Test CSE on a simple graph with redundant nodes."""
    graph = IRGraph()
    graph.nodes = {
        "n1": IRNode(id="n1", op_type="Input", inputs=[]),
        "n2": IRNode(id="n2", op_type="Add", inputs=["n1", "n1"]),
        "n3": IRNode(id="n3", op_type="Add", inputs=["n1", "n1"]),  # identical to n2
        "n4": IRNode(id="n4", op_type="Mul", inputs=["n2", "n3"]),
    }
    graph.outputs = ["n4"]

    assert cse_pass(graph) is True

    # n3 should be eliminated
    assert "n3" not in graph.nodes

    # n4's inputs should be updated to point to n2 instead of n3
    assert graph.nodes["n4"].inputs == ["n2", "n2"]


def test_cse_different_attributes():
    """Test CSE respects node attributes."""
    graph = IRGraph()
    graph.nodes = {
        "n1": IRNode(id="n1", op_type="Input", inputs=[]),
        "n2": IRNode(id="n2", op_type="Add", inputs=["n1", "n1"], attributes={"alpha": 1.0}),
        "n3": IRNode(id="n3", op_type="Add", inputs=["n1", "n1"], attributes={"alpha": 2.0}),  # different attr
    }
    graph.outputs = ["n2", "n3"]

    assert cse_pass(graph) is False
    assert "n2" in graph.nodes
    assert "n3" in graph.nodes


def test_cse_chained():
    """Test CSE can eliminate chained redundant expressions."""
    graph = IRGraph()
    graph.nodes = {
        "n1": IRNode(id="n1", op_type="Input", inputs=[]),
        "n2": IRNode(id="n2", op_type="Add", inputs=["n1", "n1"]),
        "n3": IRNode(id="n3", op_type="Add", inputs=["n1", "n1"]),  # == n2
        "n4": IRNode(id="n4", op_type="Mul", inputs=["n2", "n2"]),
        "n5": IRNode(id="n5", op_type="Mul", inputs=["n3", "n3"]),  # == n4 (since n3 == n2)
        "n6": IRNode(id="n6", op_type="Sub", inputs=["n4", "n5"]),
    }
    graph.outputs = ["n6"]

    assert cse_pass(graph) is True
    assert "n3" not in graph.nodes
    assert "n5" not in graph.nodes
    assert graph.nodes["n6"].inputs == ["n4", "n4"]


def test_cse_no_op():
    """Test CSE does nothing on graph with no common subexpressions."""
    graph = IRGraph()
    graph.nodes = {
        "n1": IRNode(id="n1", op_type="Input", inputs=[]),
        "n2": IRNode(id="n2", op_type="Input", inputs=[]),
        "n3": IRNode(id="n3", op_type="Add", inputs=["n1", "n2"]),
    }
    graph.outputs = ["n3"]
    assert cse_pass(graph) is False


def test_cse_output_update():
    """Test CSE correctly updates graph outputs."""
    graph = IRGraph()
    graph.nodes = {
        "n1": IRNode(id="n1", op_type="Input", inputs=[]),
        "n2": IRNode(id="n2", op_type="Add", inputs=["n1", "n1"]),
        "n3": IRNode(id="n3", op_type="Add", inputs=["n1", "n1"]),  # identical to n2
    }
    graph.outputs = ["n3"]

    assert cse_pass(graph) is True
    assert graph.outputs == ["n2"]
