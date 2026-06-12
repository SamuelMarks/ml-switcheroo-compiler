"""Unit tests for the constant folding optimization pass on logical graphs."""

import numpy as np
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo.transforms.passes.constant_folding import constant_folding_pass


def test_constant_folding() -> None:
    """Verifies that the constant folding pass correctly folds an 'Add' operation.

    This test constructs a logical graph with two 'Constant' nodes feeding into
    an 'Add' node. It asserts that after running the constant folding pass,
    the 'Add' node is replaced by a 'Constant' node containing the sum of the
    two inputs

    Returns:
    None
    """
    g = LogicalGraph(outputs=["n1"])
    g.nodes["c1"] = LogicalNode(
        id="c1",
        op_type="Constant",
        attributes={"value": np.array([2.0])},
    )
    g.nodes["c2"] = LogicalNode(
        id="c2",
        op_type="Constant",
        attributes={"value": np.array([3.0])},
    )
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["c1", "c2"])
    constant_folding_pass(g)
    assert g.nodes["n1"].op_type == "Constant"
    val = g.nodes["n1"].attributes["value"]
    np.testing.assert_allclose(val, np.array([5.0]))


def test_constant_folding_unsupported_op() -> None:
    """Verifies that the constant folding pass ignores unsupported operations.

    This test constructs a logical graph with an 'UnknownOp' node that has a
    'Constant' input. It asserts that the constant folding pass does not modify
    the unsupported operation

    Returns:
    None
    """
    g = LogicalGraph(outputs=["n1"])
    g.nodes["c1"] = LogicalNode(
        id="c1",
        op_type="Constant",
        attributes={"value": np.array([2.0])},
    )
    g.nodes["n1"] = LogicalNode(id="n1", op_type="UnknownOp", inputs=["c1"])
    constant_folding_pass(g)
    assert g.nodes["n1"].op_type == "UnknownOp"


def test_constant_folding_scalar_unwrap() -> None:
    """Verifies that constant folding unwraps scalar numpy arrays to Python scalars.

    This test constructs a logical graph with two integer 'Constant' nodes feeding
    into an 'Add' node. It asserts that after constant folding, the resulting
    'Constant' node's value is unwrapped from a numpy array into a standard
    Python scalar type

    Returns:
    None
    """
    g = LogicalGraph(outputs=["n1"])
    g.nodes["c1"] = LogicalNode(
        id="c1",
        op_type="Constant",
        attributes={"value": np.array([2])},
    )
    g.nodes["c2"] = LogicalNode(
        id="c2",
        op_type="Constant",
        attributes={"value": np.array([3])},
    )
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["c1", "c2"])
    constant_folding_pass(g)
    assert g.nodes["n1"].op_type == "Constant"
    assert not isinstance(g.nodes["n1"].attributes["value"], np.ndarray)
