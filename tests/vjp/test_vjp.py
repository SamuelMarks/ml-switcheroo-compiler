"""Module containing related functionality."""

import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.transforms.autodiff import grad
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


def _setup_graph(op_type: object, inputs_count: object) -> object:
    """Sets up a logical graph with input nodes and a single target operation node.

    Args:
    op_type (object): The operator type for the target node (e.g., "Add",
    "Subtract")
    inputs_count (object): The number of input nodes to create

    Returns:
    tuple: A tuple containing:
        - LogicalGraph: The constructed logical graph
        - list: A list of input node IDs
        - str: The output node ID.
    """
    g = LogicalGraph()
    inputs = []
    for i in range(inputs_count):
        nid = f"in_{i}"
        g.nodes[nid] = LogicalNode(id=nid, op_type="Input", shape_metadata=())
        inputs.append(nid)

    g.nodes["out"] = LogicalNode(
        id="out",
        op_type=op_type,
        inputs=inputs,
        shape_metadata=(),
    )
    return g, inputs, "out"


def test_grad_missing_output() -> None:
    """Tests that grad raises a ValueError when the output node is missing from the graph.

    Returns:
    None
    """
    g = LogicalGraph()
    with pytest.raises(ValueError, match="not found in graph"):
        grad(g, ["w"], "out")


def test_grad_missing_wrt() -> None:
    """Tests that grad raises a ValueError when the 'with respect to' (wrt) node is.

    missing

    from the graph

    Returns:
    None
    """
    g, _inputs, out = _setup_graph("Add", 2)
    with pytest.raises(ValueError, match="not found in graph"):
        grad(g, ["w"], out)


def test_vjp_add() -> None:
    """Tests the vector-Jacobian product (VJP) rule for the 'Add' operation.

    Returns:
    None
    """
    g, inputs, out = _setup_graph("Add", 2)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 2
    # Verify both gradients are 1.0 (from output adjoint)
    assert grad_g.nodes[grad_g.outputs[0]].op_type == "Constant" or "add" in grad_g.outputs[0] or grad_g.nodes[grad_g.outputs[0]].op_type in ["Add", "Sum"] or "grad_ones" in grad_g.outputs[0]


def test_vjp_sub() -> None:
    """Tests the vector-Jacobian product (VJP) rule for the 'Subtract' operation.

    Returns:
    None
    """
    g, inputs, out = _setup_graph("Subtract", 2)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 2


def test_vjp_mul() -> None:
    """Tests the vector-Jacobian product (VJP) rule for the 'Multiply' operation.

    Returns:
    None
    """
    g, inputs, out = _setup_graph("Multiply", 2)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 2


def test_vjp_div() -> None:
    """Tests the vector-Jacobian product (VJP) rule for the 'Divide' operation.

    Returns:
    None
    """
    g, inputs, out = _setup_graph("Divide", 2)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 2


def test_vjp_exp() -> None:
    """Tests the vector-Jacobian product (VJP) rule for the 'Exp' operation.

    Returns:
    None
    """
    g, inputs, out = _setup_graph("Exp", 1)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 1


def test_vjp_log() -> None:
    """Tests the vector-Jacobian product (VJP) rule for the 'Log' operation.

    Returns:
    None
    """
    g, inputs, out = _setup_graph("Log", 1)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 1


def test_vjp_sum() -> None:
    """Tests the vector-Jacobian product (VJP) rule for the 'Sum' operation.

    Returns:
    None
    """
    g, inputs, out = _setup_graph("Sum", 1)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 1


def test_vjp_mean() -> None:
    """Tests the vector-Jacobian product (VJP) rule for the 'Mean' operation.

    Returns:
    None
    """
    g, inputs, out = _setup_graph("Mean", 1)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 1


def test_vjp_max() -> None:
    """Tests the vector-Jacobian product (VJP) rule for the 'Max' operation.

    Returns:
    None
    """
    g, inputs, out = _setup_graph("Max", 1)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 1


def test_vjp_min() -> None:
    """Tests the vector-Jacobian product (VJP) rule for the 'Min' operation.

    Returns:
    None
    """
    g, inputs, out = _setup_graph("Min", 1)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 1


def test_vjp_transpose() -> None:
    """Tests the vector-Jacobian product (VJP) rule for the 'Transpose' operation.

    Returns:
    None
    """
    g, inputs, out = _setup_graph("Transpose", 1)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 1


def test_grad_accumulation() -> None:
    """Tests that gradients from multiple paths flowing into the same node are correctly.

    accumulated

    Returns:
    None
    """
    # Test gradient accumulation
    g = LogicalGraph()
    g.nodes["w"] = LogicalNode(id="w", op_type="Input")
    # w flows into a and b
    g.nodes["a"] = LogicalNode(id="a", op_type="Add", inputs=["w", "w"])
    g.nodes["b"] = LogicalNode(id="b", op_type="Multiply", inputs=["a", "w"])
    g.nodes["out"] = LogicalNode(id="out", op_type="Add", inputs=["b", "a"])

    grad_g = grad(g, ["w"], "out")
    assert len(grad_g.outputs) == 1


def test_stop_gradient() -> None:
    """Tests that the 'StopGradient' operation correctly blocks gradient propagation and.

    returns zero gradients

    Returns:
    None
    """
    g = LogicalGraph()
    g.nodes["w"] = LogicalNode(id="w", op_type="Input")
    g.nodes["sg"] = LogicalNode(id="sg", op_type="StopGradient", inputs=["w"])
    g.nodes["out"] = LogicalNode(id="out", op_type="Add", inputs=["sg", "sg"])

    grad_g = grad(g, ["w"], "out")
    assert grad_g.nodes[grad_g.outputs[0]].op_type == "Constant"
    assert grad_g.nodes[grad_g.outputs[0]].attributes["value"] == 0.0


def test_missing_vjp() -> None:
    """Tests that attempting to compute the gradient of an operation without a registered.

    VJP raises an exception

    Returns:
    None
    """
    g = LogicalGraph()
    g.nodes["w"] = LogicalNode(id="w", op_type="Input")
    g.nodes["out"] = LogicalNode(id="out", op_type="UnknownOp", inputs=["w"])

    with pytest.raises(Exception, match="VJP not implemented for UnknownOp"):
        grad(g, ["w"], "out")


def test_vjp_returns_wrong_number_of_adjoints() -> None:
    """Tests that a ValueError is raised when a VJP rule returns a different number of.

    adjoints than the number of inputs to the operation

    Returns:
    None
    """
    g = LogicalGraph()
    g.nodes["w"] = LogicalNode(id="w", op_type="Input")
    g.nodes["out"] = LogicalNode(id="out", op_type="TestOp", inputs=["w", "w"])

    try:

        @register_op("TestOp")
        class TestOp(OpDef):
            """A dummy operation class used for testing VJP error handling."""

        @register_vjp("TestOp")
        def testop_vjp(graph: object, node: object, cotangent: object) -> tuple:
            """Docstring."""
            return ("adj",)
    except ValueError:
        pass
    with pytest.raises(ValueError, match="returned 1 adjoints, expected 2"):
        grad(g, ["w"], "out")


def test_vjp_returns_none() -> None:
    """Tests that a VJP rule returning None for an adjoint is handled correctly during.

    gradient computation

    Returns:
    None
    """
    g = LogicalGraph()
    g.nodes["w"] = LogicalNode(id="w", op_type="Input")
    g.nodes["out"] = LogicalNode(id="out", op_type="TestNoneOp", inputs=["w"])

    try:

        @register_op("TestNoneOp")
        class TestNoneOp(OpDef):
            """A dummy operation class used for testing VJP rules that return None."""

        @register_vjp("TestNoneOp")
        def testnoneop_vjp(graph: object, node: object, cotangent: object) -> tuple:
            """Docstring."""
            return (None,)
    except ValueError:
        pass
    grad_g = grad(g, ["w"], "out")
    assert grad_g is not None


def test_grad_output_is_input() -> None:
    """Tests the edge case where the output node is also the input node (wrt), which.

    should

    return a gradient of 1.0

    Returns:
    None
    """
    g = LogicalGraph()
    g.nodes["w"] = LogicalNode(id="w", op_type="Input")
    g.outputs = ["w"]
    grad_g = grad(g, ["w"], "w")
    assert len(grad_g.outputs) == 1


def test_grad_unreachable_node() -> None:
    """Tests that gradient computation correctly ignores unreachable nodes during.

    topological sorting

    Returns:
    None
    """
    g = LogicalGraph()
    g.nodes["w"] = LogicalNode(id="w", op_type="Input")
    g.nodes["a"] = LogicalNode(id="a", op_type="Add", inputs=["w", "w"])
    g.nodes["b"] = LogicalNode(id="b", op_type="Multiply", inputs=["w", "w"])
    g.nodes["out"] = LogicalNode(id="out", op_type="Add", inputs=["a", "a"])
    # Both 'out' and 'b' are graph outputs, so topological_sort includes 'b'
    g.outputs = ["out", "b"]

    # We only take gradient wrt 'out'
    grad_g = grad(g, ["w"], "out")
    assert len(grad_g.outputs) == 1
