"""Tests for CSE."""

from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo.transforms.passes.cse import cse_pass


def test_cse() -> None:
    """Docstring."""
    g = LogicalGraph(outputs=["n3", "n4"])
    g.nodes["in"] = LogicalNode(id="in", op_type="Input")
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Relu", inputs=["in"])
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Relu", inputs=["in"])
    g.nodes["n3"] = LogicalNode(id="n3", op_type="Add", inputs=["n1", "n2"])
    g.nodes["n4"] = LogicalNode(id="n4", op_type="Add", inputs=["n2", "n1"])
    cse_pass(g)
    assert "n2" not in g.nodes
    assert g.nodes["n3"].inputs == ["n1", "n1"]


def test_cse_different_attributes() -> None:
    """Docstring."""
    g = LogicalGraph(outputs=["n1", "n2"])
    g.nodes["in"] = LogicalNode(id="in", op_type="Input")
    g.nodes["n1"] = LogicalNode(
        id="n1", op_type="Transpose", inputs=["in"], attributes={"axes": [1, 0]}
    )
    g.nodes["n2"] = LogicalNode(
        id="n2", op_type="Transpose", inputs=["in"], attributes={"axes": [0, 1]}
    )
    cse_pass(g)
    assert "n1" in g.nodes
    assert "n2" in g.nodes
