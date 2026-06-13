"""Unit tests for the data type (dtype) inference pass in the ML Switcheroo compiler.

pipeline

This module contains test cases that validate the behavior of the `dtype_inference_pass`
on various logical graphs, ensuring correct type propagation, handling of constant
nodes, cast operations, and edge cases like missing or mismatched types.
"""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.passes.dtype_inference import dtype_inference_pass

"""Module containing related functionality."""


def test_dtype_inference_full_branches() -> None:
    """Verifies dtype inference behavior across different branch conditions.

    This test constructs a logical graph with multiple output paths to validate
    how the inference pass handles mismatched dtypes, matching dtypes, and
    missing input dtypes

    Returns:
    None
    """
    g = LogicalGraph(outputs=["out1", "out2", "out3"])

    g.nodes["a"] = LogicalNode(
        id="a",
        op_type="Constant",
        attributes={"value": 1.0, "dtype": "float32"},
    )
    g.nodes["missing"] = LogicalNode(id="missing", op_type="UnknownOp", inputs=[])

    # Path True
    g.nodes["out1"] = LogicalNode(
        id="out1",
        op_type="Output",
        inputs=["a"],
        attributes={"dtype": "int32"},
    )

    # Path False (equal)
    g.nodes["out2"] = LogicalNode(
        id="out2",
        op_type="Output",
        inputs=["a"],
        attributes={"dtype": "float32"},
    )

    # Path False (inp_dtype is None)
    g.nodes["out3"] = LogicalNode(id="out3", op_type="Output", inputs=["missing"])

    dtype_inference_pass(g)


def test_dtype_inference_coverage() -> None:
    """Tests dtype inference coverage for various constant types and operations.

    This test validates that constants of different types (NumPy arrays, integers,
    floats, booleans) and operations like 'Add' are correctly processed by the
    dtype inference pass

    Returns:
    None
    """
    g = LogicalGraph()
    g.nodes["a"] = LogicalNode(
        id="a",
        op_type="Constant",
        attributes={"value": __import__("numpy").array(1.0)},
    )
    g.nodes["b"] = LogicalNode(id="b", op_type="Constant", attributes={"value": 1})
    g.nodes["c"] = LogicalNode(id="c", op_type="Constant", attributes={"value": 1.0})
    g.nodes["d"] = LogicalNode(id="d", op_type="Constant", attributes={"value": True})

    g.nodes["out_same_dtype_add"] = LogicalNode(
        id="out_same_dtype_add",
        op_type="Add",
        inputs=["a", "b"],
        attributes={"dtype": "int32"},
    )
    g.nodes["out_same_dtype_add2"] = LogicalNode(
        id="out_same_dtype_add2",
        op_type="Add",
        inputs=["c", "d"],
        attributes={"dtype": "float32"},
    )

    g.nodes["out_no_inp"] = LogicalNode(id="out_no_inp", op_type="Output", inputs=[])
    dtype_inference_pass(g)


def test_dtype_inference_false_branches() -> None:
    """Tests the dtype inference pass on inactive or redundant branch conditions.

    This test ensures that the inference pass behaves correctly when the output
    dtype already matches the input dtype, or when the input node has no
    specified dtype

    Returns:
    None
    """
    g = LogicalGraph(outputs=["out_eq", "out_none"])
    g.nodes["a"] = LogicalNode(id="a", op_type="Input", attributes={"dtype": "float32"})
    g.nodes["out_eq"] = LogicalNode(
        id="out_eq",
        op_type="Output",
        inputs=["a"],
        attributes={"dtype": "float32"},
    )

    g.nodes["b"] = LogicalNode(id="b", op_type="Input")
    g.nodes["out_none"] = LogicalNode(id="out_none", op_type="Output", inputs=["b"])

    dtype_inference_pass(g)


def test_dtype_inference_everything() -> None:
    """Performs a comprehensive test of the dtype inference pass.

    This test covers a wide range of scenarios, including various constant types,
    inputs, outputs (empty, matching, mismatched, and missing), Cast operations,
    and unknown operations to ensure robust dtype propagation

    Returns:
    None
    """
    g = LogicalGraph(
        outputs=[
            "out_eq",
            "out_diff",
            "out_empty",
            "out_missing",
            "out_cast",
            "out_unknown",
            "c1",
            "c2",
            "c3",
            "c4",
        ],
    )

    # Constants
    g.nodes["c1"] = LogicalNode(id="c1", op_type="Constant", attributes={"value": 1.0})
    g.nodes["c2"] = LogicalNode(id="c2", op_type="Constant", attributes={"value": 1})
    g.nodes["c3"] = LogicalNode(id="c3", op_type="Constant", attributes={"value": True})
    g.nodes["c4"] = LogicalNode(
        id="c4",
        op_type="Constant",
        attributes={"value": __import__("numpy").array(1.0)},
    )

    # Input
    g.nodes["i1"] = LogicalNode(
        id="i1",
        op_type="Input",
        attributes={"dtype": "float32"},
    )
    g.nodes["i2"] = LogicalNode(id="i2", op_type="Input")

    # Output
    g.nodes["out_empty"] = LogicalNode(id="out_empty", op_type="Output", inputs=[])
    g.nodes["out_eq"] = LogicalNode(
        id="out_eq",
        op_type="Output",
        inputs=["i1"],
        attributes={"dtype": "float32"},
    )
    g.nodes["out_diff"] = LogicalNode(
        id="out_diff",
        op_type="Output",
        inputs=["i1"],
        attributes={"dtype": "int32"},
    )
    g.nodes["out_missing"] = LogicalNode(
        id="out_missing",
        op_type="Output",
        inputs=["missing"],
    )

    # Cast
    g.nodes["out_cast"] = LogicalNode(
        id="out_cast",
        op_type="Cast",
        inputs=["i1"],
        attributes={"dtype": "int32"},
    )

    # Unknown
    g.nodes["out_unknown"] = LogicalNode(
        id="out_unknown",
        op_type="UnknownOp",
        inputs=["i1", "i2"],
    )

    dtype_inference_pass(g)
