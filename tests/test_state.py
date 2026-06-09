"""Tests for state functionalization."""

from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo_compiler.state import lift_state


def test_lift_state():
    """Docstring."""
    g = LogicalGraph(outputs=["update2"])

    # Initial state
    g.nodes["s1"] = LogicalNode(id="s1", op_type="Input")

    # Computation
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["s1", "s1"])

    # State update
    g.nodes["update1"] = LogicalNode(
        id="update1", op_type="Assign", inputs=["s1", "n1"]
    )

    # Computation using updated state
    g.nodes["update2"] = LogicalNode(
        id="update2", op_type="Mul", inputs=["s1", "update1"]
    )

    func_g = lift_state(g, ["s1"])

    # "update1" should be eliminated
    assert "update1" not in func_g.nodes

    # n1 reads initial state
    assert func_g.nodes["n1"].inputs == ["s1", "s1"]

    # n2 reads updated state "n1"
    assert func_g.nodes["update2"].inputs == ["n1", "update1"]

    # Outputs should be original output + updated state
    assert func_g.outputs == ["update2", "n1"]


def test_lift_state_no_update():
    """Docstring."""
    g = LogicalGraph(outputs=["n1"])
    g.nodes["s1"] = LogicalNode(id="s1", op_type="Input")
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["s1", "s1"])

    func_g = lift_state(g, ["s1"])

    assert func_g.outputs == ["n1", "s1"]


def test_lift_state_assign_non_state():
    # If assign targets a non-state variable, it's ignored or passed through
    """Docstring."""
    g = LogicalGraph(outputs=["n1"])
    g.nodes["s1"] = LogicalNode(id="s1", op_type="Input")
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["s1", "s1"])
    g.nodes["update1"] = LogicalNode(
        id="update1", op_type="Assign", inputs=["n1", "s1"]
    )

    func_g = lift_state(g, ["s1"])

    assert "update1" not in func_g.nodes
    assert func_g.outputs == ["n1", "s1"]
