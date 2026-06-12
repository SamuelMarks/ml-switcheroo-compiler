"""Tests for lift state pass."""

from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo.transforms.passes.lift_state import lift_state_pass


def test_lift_state_pass() -> None:
    """Docstring."""
    g = LogicalGraph(outputs=["n1"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="ReadVariable")
    g.nodes["n2"] = LogicalNode(id="n2", op_type="AssignVariable", inputs=["n1"])

    assert lift_state_pass(g) is True
    assert g.nodes["n1"].op_type == "Input"
    assert g.nodes["n2"].op_type == "Output"
    assert "n2" in g.outputs

    assert lift_state_pass(g) is False
