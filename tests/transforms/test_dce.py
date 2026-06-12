"""Tests for DCE."""

from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo.transforms.passes.dce import dce_pass


def test_dce() -> None:
    """Docstring."""
    g = LogicalGraph(outputs=["n2"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=[])
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Add", inputs=[])
    g.nodes["n3"] = LogicalNode(id="n3", op_type="Add", inputs=["n1"])
    dce_pass(g)
    assert "n1" not in g.nodes
    assert "n3" not in g.nodes
    assert "n2" in g.nodes


def test_dce_chained() -> None:
    """Docstring."""
    g = LogicalGraph(outputs=["n1"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Input")
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Add", inputs=["n1"])
    g.nodes["n3"] = LogicalNode(id="n3", op_type="Add", inputs=["n2"])
    g.nodes["n4"] = LogicalNode(id="n4", op_type="Add", inputs=["n3"])
    dce_pass(g)
    assert "n2" not in g.nodes
    assert "n3" not in g.nodes
    assert "n4" not in g.nodes
    assert "n1" in g.nodes
