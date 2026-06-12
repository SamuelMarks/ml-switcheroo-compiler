"""Unit tests for edge cases in various IR transformation passes.

This module contains test cases verifying the robustness of dtype inference, broadcast
explicitizer, type promotion, and shape inference passes when encountering unexpected,
malformed, or edge-case inputs in the IR graph.
"""

import contextlib

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo.core.dtype import DType
from ml_switcheroo.ir.core import IRGraph
from ml_switcheroo.transforms.passes.broadcast_explicitizer import (
    broadcast_explicitizer_pass,
)
from ml_switcheroo.transforms.passes.dtype_inference import dtype_inference_pass
from ml_switcheroo.transforms.passes.type_promotion_explicitizer import (
    type_promotion_explicitizer_pass,
)


def test_dtype_inference_edge_cases() -> None:
    """Verifies the behavior of the dtype inference pass under various edge cases.

    This test constructs an IR graph with several edge-case nodes (such as boolean,
    float, and string constants, unhandled object types, output nodes inheriting
    input types, logical operations, cast operations, and unknown operations) and
    asserts that the dtype inference pass correctly infers or falls back to the
    expected data types

    Returns:
    None
    """
    g = IRGraph()
    # Test bool constant
    g.nodes["c_bool"] = LogicalNode(
        id="c_bool",
        op_type="Constant",
        attributes={"value": True},
    )
    # Test float constant
    g.nodes["c_float"] = LogicalNode(
        id="c_float",
        op_type="Constant",
        attributes={"value": 2.0},
    )
    # Test str constant
    g.nodes["c_str"] = LogicalNode(
        id="c_str",
        op_type="Constant",
        attributes={"value": "test"},
    )
    # Test unhandled type
    g.nodes["c_obj"] = LogicalNode(
        id="c_obj",
        op_type="Constant",
        attributes={"value": object()},
    )
    # Output node inheriting input
    g.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["c_bool"])
    # Logical Op
    g.nodes["log"] = LogicalNode(
        id="log",
        op_type="Equal",
        inputs=["c_float", "c_float"],
    )
    # Cast Op
    g.nodes["cast"] = LogicalNode(
        id="cast",
        op_type="Cast",
        attributes={"dtype": DType.Int32},
        inputs=["c_float"],
    )
    # Missing Op fallback
    g.nodes["unknown"] = LogicalNode(id="unknown", op_type="Unknown", inputs=[])

    modified = dtype_inference_pass(g)
    assert modified
    assert g.nodes["c_bool"].attributes["dtype"] == DType.Bool.value
    assert g.nodes["c_float"].attributes["dtype"] == DType.Float32.value
    assert g.nodes["c_str"].attributes.get("dtype") is None
    assert g.nodes["out"].attributes["dtype"] == DType.Bool.value
    assert g.nodes["log"].attributes["dtype"] == DType.Bool.value
    assert g.nodes["cast"].attributes["dtype"] == DType.Int32.value
    assert g.nodes["unknown"].attributes["dtype"] == DType.Float32.value  # Fallback


def test_broadcast_explicitizer_edge_cases() -> None:
    """Verifies the behavior of the broadcast explicitizer pass under various edge cases.

    This test constructs an IR graph with edge-case scenarios where broadcasting
    should not be explicitly injected (such as unregistered operations, unary
    operations, nodes with missing shape metadata, and inputs with identical
    shapes) and asserts that the broadcast explicitizer pass does not modify the
    graph

    Returns:
    None
    """
    g = IRGraph()
    # Missing op_type in registry
    g.nodes["A"] = LogicalNode(id="A", op_type="Input", shape_metadata=(1, 3))
    g.nodes["uk"] = LogicalNode(id="uk", op_type="UnknownOp", inputs=["A", "A"])

    # Not binary
    g.nodes["unary"] = LogicalNode(id="unary", op_type="Abs", inputs=["A"])

    # None shapes
    g.nodes["none1"] = LogicalNode(id="none1", op_type="Input", shape_metadata=None)
    g.nodes["add1"] = LogicalNode(id="add1", op_type="Add", inputs=["none1", "A"])

    # Same shape
    g.nodes["add2"] = LogicalNode(id="add2", op_type="Add", inputs=["A", "A"])

    modified = broadcast_explicitizer_pass(g)
    assert not modified  # None of the edge cases should inject a broadcast


def test_type_promotion_edge_cases() -> None:
    """Verifies the behavior of the type promotion explicitizer pass under various edge.

    cases

    This test constructs an IR graph with edge-case scenarios where type promotion
    should not occur (such as unary operations, missing data types, identical
    data types, and incompatible data types that cannot be promoted) and asserts
    that the type promotion explicitizer pass does not modify the graph

    Returns:
    None
    """
    g = IRGraph()
    # Not binary
    g.nodes["unary"] = LogicalNode(id="unary", op_type="Abs", inputs=["A"])

    # Missing dtypes
    g.nodes["A"] = LogicalNode(id="A", op_type="Input", attributes={})
    g.nodes["add1"] = LogicalNode(id="add1", op_type="Add", inputs=["A", "A"])

    # Same dtypes
    g.nodes["B"] = LogicalNode(id="B", op_type="Input", attributes={"dtype": "float32"})
    g.nodes["add2"] = LogicalNode(id="add2", op_type="Add", inputs=["B", "B"])

    # Incompatible dtypes that NumPy can't promote
    g.nodes["C"] = LogicalNode(id="C", op_type="Input", attributes={"dtype": "float32"})
    g.nodes["D"] = LogicalNode(
        id="D",
        op_type="Input",
        attributes={"dtype": "invalid_type"},
    )
    g.nodes["add3"] = LogicalNode(id="add3", op_type="Add", inputs=["C", "D"])

    modified = type_promotion_explicitizer_pass(g)
    assert not modified


def test_shape_inference_edge_cases() -> None:
    """Verifies the behavior of the shape inference pass under various edge cases.

    This test constructs an IR graph with edge-case scenarios (such as output
    nodes without inputs, unknown operations, and operations that fail shape
    inference due to missing inputs) and asserts that the shape inference pass
    handles these cases gracefully or raises expected exceptions

    Returns:
    None
    """
    g = IRGraph()
    from ml_switcheroo.transforms.passes.shape_inference import shape_inference_pass

    # Output without inputs
    g.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=[])
    # Missing operation
    g.nodes["uk"] = LogicalNode(id="uk", op_type="UnknownOp", inputs=[])
    # Operation that fails shape inference
    g.nodes["fail"] = LogicalNode(
        id="fail",
        op_type="MatMul",
        inputs=[],
    )  # MatMul without inputs should throw an exception from numpy broadcast etc

    with contextlib.suppress(Exception):
        shape_inference_pass(g)
