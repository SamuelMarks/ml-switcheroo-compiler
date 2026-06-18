"""Unit tests for the state lifting functionality and state operations.

This module contains tests for `lift_state` which transforms a stateful logical graph
into a stateless functional graph by lifting state variables to inputs and outputs. It
also tests the behavior of state-related operations like ReadVariable and
AssignVariable.
"""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.state_manager import lift_state


def test_lift_state() -> None:
    """Tests the `lift_state` function with a graph containing state updates.

    This test verifies that state updates (Assign operations) are correctly tracked,
    that Assign nodes are eliminated from the graph, and that the graph outputs
    are updated to include the final state values

    Returns:
    None
    """
    g = LogicalGraph(outputs=["update2"])

    # Initial state
    g.nodes["s1"] = LogicalNode(id="s1", op_type="Input")

    # Computation
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["s1", "s1"])

    # State update
    g.nodes["update1"] = LogicalNode(
        id="update1",
        op_type="Assign",
        inputs=["s1", "n1"],
    )

    # Computation using updated state
    g.nodes["update2"] = LogicalNode(
        id="update2",
        op_type="Mul",
        inputs=["s1", "update1"],
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


def test_lift_state_no_update() -> None:
    """Tests the `lift_state` function when there are no state updates in the graph.

    This test verifies that if a state variable is declared but never updated via
    an Assign operation, the original state is correctly passed through to the
    outputs

    Returns:
    None
    """
    g = LogicalGraph(outputs=["n1"])
    g.nodes["s1"] = LogicalNode(id="s1", op_type="Input")
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["s1", "s1"])

    func_g = lift_state(g, ["s1"])

    assert func_g.outputs == ["n1", "s1"]


def test_lift_state_assign_non_state() -> None:
    """Tests `lift_state` when an Assign operation targets a non-state variable.

    This test verifies that Assign operations targeting variables not declared as
    state variables are ignored or eliminated, and do not affect the state lifting
    process

    Returns:
    None
    """
    # If assign targets a non-state variable, it's ignored or passed through
    g = LogicalGraph(outputs=["n1"])
    g.nodes["s1"] = LogicalNode(id="s1", op_type="Input")
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["s1", "s1"])
    g.nodes["update1"] = LogicalNode(
        id="update1",
        op_type="Assign",
        inputs=["n1", "s1"],
    )

    func_g = lift_state(g, ["s1"])

    assert "update1" not in func_g.nodes
    assert func_g.outputs == ["n1", "s1"]


def test_lift_state_assign_not_in_env() -> None:
    """Tests `lift_state` when an Assign operation targets a variable not in the.

    environment

    This test verifies that Assign nodes targeting variables that are not part of
    the
    state variables list are dropped from the resulting graph

    Returns:
    None
    """
    g = LogicalGraph()
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Input")
    g.nodes["a"] = LogicalNode(id="a", op_type="Assign", inputs=["n1", "n1"])
    g.outputs = ["a"]

    out_g = lift_state(g, state_vars=[])
    # The Assign node should be dropped and target not in env
    assert "a" not in out_g.nodes


def test_state_ops() -> None:
    """Tests the behavior of state operations like ReadVariable and AssignVariable.

    This test verifies that shape inference works correctly for these operations
    and that attempting to evaluate them using numpy raises a CompilationError

    Returns:
    None
    """
    import pytest

    from ml_switcheroo_compiler.core.errors import CompilationError
    from ml_switcheroo_compiler.ops.base import get_op

    r = get_op("ReadVariable")()
    a = get_op("AssignVariable")()

    assert r.infer_shape(shape=(2,)) == (2,)
    assert a.infer_shape((2,)) == (2,)

    with pytest.raises(CompilationError):
        r.eager_eval()

    with pytest.raises(CompilationError):
        a.eager_eval(1)


def test_keras_batch_normalization_state_lifting() -> None:
    """Validate that state lifting supports Keras moving statistics."""
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.state_lifting import state_lifting_pass

    graph = IRGraph()
    # Initial read of moving mean
    graph.nodes["read_mean"] = IRNode(
        "read_mean", "ReadVariable", attributes={"variable_name": "moving_mean"}
    )
    # Some computation
    graph.nodes["new_mean"] = IRNode("new_mean", "Add", inputs=["read_mean"])
    # Assignment back to moving mean
    graph.nodes["assign_mean"] = IRNode(
        "assign_mean",
        "AssignVariable",
        inputs=["new_mean"],
        attributes={"variable_name": "moving_mean"},
    )

    # Another output representing standard output
    graph.nodes["output"] = IRNode("output", "Identity", inputs=["new_mean"])
    graph.outputs = ["output"]

    modified = state_lifting_pass(graph)

    assert modified
    # ReadVariable should become an Input node
    assert graph.nodes["read_mean"].op_type == "Input"
    assert graph.nodes["read_mean"].attributes["name"] == "moving_mean"

    # AssignVariable should become an Output node
    assert graph.nodes["assign_mean"].op_type == "Output"
    assert graph.nodes["assign_mean"].attributes["name"] == "moving_mean_out"
    assert "assign_mean" in graph.outputs
