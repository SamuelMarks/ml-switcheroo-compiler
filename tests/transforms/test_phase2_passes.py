"""Unit tests for intermediate representation (IR) graph transformation passes.

This module contains test cases verifying the correctness of various compiler passes,
including data type inference, explicit broadcasting, type promotion, and state lifting
on logical IR graphs.
"""

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo.core.dtype import DType
from ml_switcheroo.ir.core import IRGraph
from ml_switcheroo.transforms.passes.broadcast_explicitizer import (
    broadcast_explicitizer_pass,
)
from ml_switcheroo.transforms.passes.dtype_inference import dtype_inference_pass
from ml_switcheroo.transforms.passes.state_lifting import state_lifting_pass
from ml_switcheroo.transforms.passes.type_promotion_explicitizer import (
    type_promotion_explicitizer_pass,
)


def test_dtype_inference() -> None:
    """Verifies that the dtype inference pass correctly propagates and infers data types.

    This test constructs a graph with a constant node (implicitly float32), an input
    node
    with an explicit int32 dtype, and an addition node. It asserts that the pass
    infers
    the constant's dtype and promotes the addition's output dtype to float64

    Returns:
    None
    """
    g = IRGraph()
    g.nodes["c"] = LogicalNode(
        id="c",
        op_type="Constant",
        attributes={"value": 1.0},
    )  # Should be float32
    g.nodes["i"] = LogicalNode(
        id="i",
        op_type="Input",
        attributes={"dtype": DType.Int32.value},
    )
    g.nodes["add"] = LogicalNode(id="add", op_type="Add", inputs=["c", "i"])
    g.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["add"])

    modified = dtype_inference_pass(g)
    assert modified
    assert g.nodes["c"].attributes["dtype"] == DType.Float32.value
    assert (
        g.nodes["add"].attributes["dtype"] == "float64"
    )  # promote_types(float32, int32)


def test_broadcast_explicitizer() -> None:
    """Verifies that the broadcast explicitizer pass inserts explicit broadcast nodes.

    This test constructs a graph with two inputs of mismatched but compatible shapes
    ((1, 3) and (2, 3)) fed into an addition node. It asserts that the pass inserts
    a
    'BroadcastTo' node for the smaller input to match the target shape

    Returns:
    None
    """
    g = IRGraph()
    g.nodes["A"] = LogicalNode(id="A", op_type="Input", shape_metadata=(1, 3))
    g.nodes["B"] = LogicalNode(id="B", op_type="Input", shape_metadata=(2, 3))
    g.nodes["add"] = LogicalNode(id="add", op_type="Add", inputs=["A", "B"])

    modified = broadcast_explicitizer_pass(g)
    assert modified
    # A should be explicitly broadcasted to (2, 3)
    in1 = g.nodes["add"].inputs[0]
    in2 = g.nodes["add"].inputs[1]

    assert g.nodes[in1].op_type == "BroadcastTo"
    assert g.nodes[in1].shape_metadata == (2, 3)
    assert in2 == "B"  # B is already (2,3)


def test_type_promotion() -> None:
    """Verifies that the type promotion pass inserts explicit cast nodes for mismatched.

    types

    This test constructs a graph with float32 and int32 inputs fed into an addition
    node
    It asserts that the pass inserts explicit 'Cast' nodes to promote both inputs to
    float64 according to type promotion rules

    Returns:
    None
    """
    g = IRGraph()
    g.nodes["A"] = LogicalNode(id="A", op_type="Input", attributes={"dtype": "float32"})
    g.nodes["B"] = LogicalNode(id="B", op_type="Input", attributes={"dtype": "int32"})
    g.nodes["add"] = LogicalNode(id="add", op_type="Add", inputs=["A", "B"])

    modified = type_promotion_explicitizer_pass(g)
    assert modified

    in1 = g.nodes["add"].inputs[0]
    in2 = g.nodes["add"].inputs[1]

    # Both should be cast to float64, A might be cast depending on numpy rules
    # float32 + int32 -> float64
    assert g.nodes[in1].op_type == "Cast"
    assert g.nodes[in1].attributes["dtype"] == "float64"
    assert g.nodes[in2].op_type == "Cast"
    assert g.nodes[in2].attributes["dtype"] == "float64"


def test_state_lifting() -> None:
    """Verifies that the state lifting pass converts stateful operations to inputs and.

    outputs

    This test constructs a graph with 'ReadVariable' and 'AssignVariable' nodes. It
    asserts
    that the pass successfully lifts these stateful operations, converting the read
    to an
    'Input' node and the assign to an 'Output' node

    Returns:
    None
    """
    g = IRGraph()
    g.nodes["r"] = LogicalNode(
        id="r",
        op_type="ReadVariable",
        attributes={"variable_name": "my_var"},
    )
    g.nodes["add"] = LogicalNode(id="add", op_type="Add", inputs=["r", "r"])
    g.nodes["a"] = LogicalNode(
        id="a",
        op_type="AssignVariable",
        inputs=["add"],
        attributes={"variable_name": "my_var"},
    )

    modified = state_lifting_pass(g)
    assert modified
    assert g.nodes["r"].op_type == "Input"
    assert g.nodes["r"].attributes["name"] == "my_var"
    assert g.nodes["a"].op_type == "Output"
    assert "a" in g.outputs


def test_type_promotion_partial_branches() -> None:
    """Verifies type promotion when only one branch requires casting to the target type.

    This test constructs a graph with multiple addition branches where one input
    matches
    the target promoted type (float32) and the other does not (int32). It ensures
    that
    the type promotion pass correctly handles partial casting scenarios

    Returns:
    None
    """
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo.transforms.passes.type_promotion_explicitizer import (
        type_promotion_explicitizer_pass,
    )

    g = LogicalGraph(outputs=["add1", "add2"])

    # dt1 == target, dt2 != target
    g.nodes["a1"] = LogicalNode(
        id="a1",
        op_type="Input",
        attributes={"dtype": "float32"},
    )
    g.nodes["b1"] = LogicalNode(id="b1", op_type="Input", attributes={"dtype": "int32"})
    g.nodes["add1"] = LogicalNode(id="add1", op_type="Add", inputs=["a1", "b1"])

    # dt1 != target, dt2 == target
    g.nodes["a2"] = LogicalNode(id="a2", op_type="Input", attributes={"dtype": "int32"})
    g.nodes["b2"] = LogicalNode(
        id="b2",
        op_type="Input",
        attributes={"dtype": "float32"},
    )
    g.nodes["add2"] = LogicalNode(id="add2", op_type="Add", inputs=["a2", "b2"])

    type_promotion_explicitizer_pass(g)
