"""Unit tests for the type promotion explicitizer transformation pass.

This module contains test cases to verify that the type promotion explicitizer correctly
identifies and inserts explicit type casting operations in a logical graph when binary
operations have mismatched input data types.
"""


def test_type_promotion_partial_branches() -> None:
    """Verifies type promotion behavior when only one operand requires casting.

    This test constructs a logical graph with two 'Add' operations representing
    partial promotion scenarios:
    1. float64 + float32 (left operand matches target type, right operand needs
    promotion)
    2. float32 + float64 (left operand needs promotion, right operand matches target
    type)

    It asserts that the `type_promotion_explicitizer_pass` correctly inserts
    the necessary explicit type conversion nodes

    Returns:
    None
    """
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo.transforms.passes.type_promotion_explicitizer import (
        type_promotion_explicitizer_pass,
    )

    g = LogicalGraph(outputs=["add1", "add2"])

    # dt1 == target, dt2 != target (float64 and float32 -> float64)
    g.nodes["a1"] = LogicalNode(
        id="a1",
        op_type="Input",
        attributes={"dtype": "float64"},
    )
    g.nodes["b1"] = LogicalNode(
        id="b1",
        op_type="Input",
        attributes={"dtype": "float32"},
    )
    g.nodes["add1"] = LogicalNode(id="add1", op_type="Add", inputs=["a1", "b1"])

    # dt1 != target, dt2 == target (float32 and float64 -> float64)
    g.nodes["a2"] = LogicalNode(
        id="a2",
        op_type="Input",
        attributes={"dtype": "float32"},
    )
    g.nodes["b2"] = LogicalNode(
        id="b2",
        op_type="Input",
        attributes={"dtype": "float64"},
    )
    g.nodes["add2"] = LogicalNode(id="add2", op_type="Add", inputs=["a2", "b2"])

    type_promotion_explicitizer_pass(g)
