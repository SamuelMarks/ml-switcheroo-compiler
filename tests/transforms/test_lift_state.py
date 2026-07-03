"""Unit tests for the state lifting transformation passes.

This module contains test cases to verify the correctness of the state lifting passes,
which convert stateful operations (such as variable reads and writes) into explicit
graph inputs and outputs.
"""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.passes.lift_state import lift_state_pass
from ml_switcheroo_compiler.transforms.passes.state_lifting import state_lifting_pass


def test_lift_state_pass() -> None:
    """Verifies that the lift state pass correctly transforms stateful nodes.

    This test constructs a logical graph with 'ReadVariable' and 'AssignVariable'
    nodes, applies the `lift_state_pass`, and asserts that:
    1. The pass returns True, indicating a modification was made
    2. 'ReadVariable' is converted to an 'Input' node
    3. 'AssignVariable' is converted to an 'Output' node and added to the graph's
    outputs
    4. Running the pass a second time returns False, indicating no further changes

    Returns:
    None
    """
    g = LogicalGraph(outputs=["n1"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="ReadVariable")
    g.nodes["n2"] = LogicalNode(id="n2", op_type="AssignVariable", inputs=["n1"])

    assert lift_state_pass(g) is True
    assert g.nodes["n1"].op_type == "Input"
    assert g.nodes["n2"].op_type == "Output"
    assert "n2" in g.outputs

    assert lift_state_pass(g) is False


def test_state_lifting_already_in_outputs() -> None:
    """Verifies state lifting behavior when the target node is already in the graph.

    outputs

    This test constructs a logical graph where an 'AssignVariable' node is already
    listed in the graph's outputs, applies the `state_lifting_pass`, and ensures
    that the pass executes successfully without errors

    Returns:
    None
    """
    g = LogicalGraph(outputs=["n1"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="AssignVariable", inputs=["a", "b"])
    state_lifting_pass(g)
