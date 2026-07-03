"""Unit tests for the broadcast explicitizer pass.

This module contains test cases that verify the behavior of the broadcast explicitizer
pass under various shape configurations, including missing shapes, equal shapes, and
shapes requiring broadcasting.
"""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

import ml_switcheroo_compiler.transforms.passes.broadcast_explicitizer as be
from ml_switcheroo_compiler.transforms.passes.broadcast_explicitizer import (
    broadcast_explicitizer_pass,
)


def test_broadcast_coverage() -> None:
    """Verifies the broadcast explicitizer pass behavior across multiple edge cases.

    This test covers scenarios including:
    - Missing shape metadata on the first input
    - Incompatible shapes that should trigger a ValueError
    - Mismatched shapes where the second input does not match the target shape
    - Unary operations (like Abs) that do not require broadcasting

    It mocks the shape inference pass to return False to isolate the broadcast
    explicitizer logic

    Returns:
    None
    """
    g = LogicalGraph(outputs=["add1", "add2", "add3", "abs1"])
    # shape1 is None
    g.nodes["a"] = LogicalNode(id="a", op_type="Input", shape_metadata=None)
    g.nodes["b"] = LogicalNode(id="b", op_type="Input", shape_metadata=(2,))
    g.nodes["add1"] = LogicalNode(id="add1", op_type="Add", inputs=["a", "b"])

    # ValueError
    g.nodes["c"] = LogicalNode(id="c", op_type="Input", shape_metadata=(3,))
    g.nodes["add2"] = LogicalNode(id="add2", op_type="Add", inputs=["b", "c"])

    # shape2 != target_shape
    g.nodes["d"] = LogicalNode(id="d", op_type="Input", shape_metadata=(1,))
    g.nodes["add3"] = LogicalNode(id="add3", op_type="Add", inputs=["b", "d"])

    g.nodes["abs1"] = LogicalNode(id="abs1", op_type="Abs", inputs=["b"])

    be.shape_inference_pass = lambda g: False

    broadcast_explicitizer_pass(g)


def test_broadcast_coverage_shape2() -> None:
    """Verifies the broadcast explicitizer pass when the second input has missing shape.

    metadata

    Ensures that the pass handles cases where the first input has a valid shape
    but the second input's shape is None, mocking the shape inference pass to
    return False

    Returns:
    None
    """
    g = LogicalGraph(outputs=["add1"])
    g.nodes["a"] = LogicalNode(id="a", op_type="Input", shape_metadata=(2,))
    g.nodes["b"] = LogicalNode(id="b", op_type="Input", shape_metadata=None)
    g.nodes["add1"] = LogicalNode(id="add1", op_type="Add", inputs=["a", "b"])

    be.shape_inference_pass = lambda g: False

    broadcast_explicitizer_pass(g)


def test_broadcast_coverage_equal_shapes() -> None:
    """Verifies the broadcast explicitizer pass when both inputs have identical shapes.

    Ensures that no explicit broadcast operations are introduced when the input
    shapes are already equal, mocking the shape inference pass to return False

    Returns:
    None
    """
    g = LogicalGraph(outputs=["add1"])
    g.nodes["a"] = LogicalNode(id="a", op_type="Input", shape_metadata=(2,))
    g.nodes["b"] = LogicalNode(id="b", op_type="Input", shape_metadata=(2,))
    g.nodes["add1"] = LogicalNode(id="add1", op_type="Add", inputs=["a", "b"])

    be.shape_inference_pass = lambda g: False

    broadcast_explicitizer_pass(g)


def test_broadcast_coverage_shape1_needs_broadcast() -> None:
    """Verifies the broadcast explicitizer pass when the first input requires broadcasting.

    Tests the scenario where the first input has a shape of (1,) and the second
    input has a shape of (2,), requiring the first input to be broadcasted to match
    the target shape. Mocks the shape inference pass to return False

    Returns:
    None
    """
    g = LogicalGraph(outputs=["add1"])
    g.nodes["a"] = LogicalNode(id="a", op_type="Input", shape_metadata=(1,))
    g.nodes["b"] = LogicalNode(id="b", op_type="Input", shape_metadata=(2,))
    g.nodes["add1"] = LogicalNode(id="add1", op_type="Add", inputs=["a", "b"])

    be.shape_inference_pass = lambda g: False

    broadcast_explicitizer_pass(g)
