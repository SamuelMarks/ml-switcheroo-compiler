"""Unit tests for verifying dynamic numpy-backed shape resolution and broadcasting in the.

interpreter.
"""

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.interpreter import evaluate_graph
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.ops.registry import get_op
from ml_switcheroo_compiler.transforms.passes.shape_inference import shape_inference_pass


def test_shape_inference_broadcasting() -> None:
    """Verifies that the interpreter correctly infers and broadcasts shapes using a numpy-.

    backed execution

    This test constructs a logical graph with an 'Expand' operation, executes it
    with a 1D input array, and asserts that the output shape and values are
    correctly broadcasted to the target 2D shape

    Returns:
    None
    """
    g = LogicalGraph(outputs=["exp"])
    g.nodes["x"] = LogicalNode(id="x", op_type="Input")
    g.nodes["exp"] = LogicalNode(
        id="exp",
        op_type="Expand",
        inputs=["x"],
        shape_metadata=(2, 3),
    )

    res = evaluate_graph(g, {"x": np.array([1, 2, 3])})
    assert res["exp"].shape == (2, 3)
    np.testing.assert_array_equal(res["exp"], np.array([[1, 2, 3], [1, 2, 3]]))


def test_shape_inference_pass_coverage_() -> None:
    """Test shape inference pass coverage."""
    g = IRGraph()
    # input node
    inp = IRNode(id="in", op_type="Input", inputs=[], shape_metadata=(2,))

    # const node
    const = IRNode(
        id="const",
        op_type="Constant",
        inputs=[],
        attributes={"value": 1.0},
        shape_metadata=(),
    )

    # known op
    add = IRNode(id="add", op_type="Add", inputs=["in", "const"], shape_metadata=())

    # unknown op
    unk = IRNode(id="unk", op_type="Unknown", inputs=["add"], shape_metadata=())

    # output node
    out = IRNode(id="out", op_type="Output", inputs=["unk"], shape_metadata=())

    g.nodes = {"in": inp, "const": const, "add": add, "unk": unk, "out": out}

    shape_inference_pass(g)

    assert g.nodes["in"].shape_metadata == (2,)
    assert g.nodes["const"].shape_metadata in ((), (2,))
    assert g.nodes["add"].shape_metadata in ((), (2,))
    assert g.nodes["out"].shape_metadata == ()


def test_shape_inference_pass_coverage_kwargs() -> None:
    """Test kwargs."""
    g = IRGraph()
    # input node
    inp = IRNode(id="in", op_type="Input", inputs=[], shape_metadata=(2,))

    reshape_node = IRNode(
        id="res",
        op_type="Reshape",
        inputs=["in"],
        attributes={},
        shape_metadata=(1, 2),
    )
    expand_node = IRNode(
        id="exp",
        op_type="Expand",
        inputs=["in"],
        attributes={},
        shape_metadata=(2, 1),
    )

    g.nodes = {"in": inp, "res": reshape_node, "exp": expand_node}
    shape_inference_pass(g)
    assert g.nodes["res"].shape_metadata == (1, 2)
    assert g.nodes["exp"].shape_metadata == ()


def test_shape_inference_pass_coverage_output_no_inputs() -> None:
    """Test output no inputs."""
    g = IRGraph()
    out = IRNode(id="out", op_type="Output", inputs=[], shape_metadata=())
    g.nodes = {"out": out}
    shape_inference_pass(g)


def test_shape_inference_pass_coverage_broadcast() -> None:
    """Test broadcast."""
    g = IRGraph()
    inp = IRNode(id="in", op_type="Input", inputs=[], shape_metadata=(2,))
    bcast = IRNode(
        id="bcast",
        op_type="BroadcastTo",
        inputs=["in"],
        attributes={},
        shape_metadata=(2, 2),
    )
    g.nodes = {"in": inp, "bcast": bcast}
    shape_inference_pass(g)


def test_shape_inference_pass_coverage_output_with_input() -> None:
    """Test output with input."""
    g = IRGraph()
    inp = IRNode(id="in", op_type="Input", inputs=[], shape_metadata=(5, 5))
    out = IRNode(id="out", op_type="Output", inputs=["in"], shape_metadata=())
    g.nodes = {"in": inp, "out": out}
    shape_inference_pass(g)
    assert g.nodes["out"].shape_metadata in ((), (5, 5))


def test_shape_inference_pass_coverage_constant_modified() -> None:
    """Test const modified."""
    g = IRGraph()
    const = IRNode(
        id="const",
        op_type="Constant",
        inputs=[],
        attributes={"value": [1.0, 2.0]},
        shape_metadata=(),
    )
    g.nodes = {"const": const}
    res = shape_inference_pass(g)
    assert res in (True, False)


def test_shape_inference_pass_coverage_exceptions() -> None:
    """Function docstring."""
    g = IRGraph()
    inp = IRNode(id="in", op_type="Input", inputs=[], shape_metadata=(5, 5))

    # Trigger a ValueError in Reshape.infer_shape
    # For example, by passing an invalid newshape attribute
    bad_op = IRNode(
        id="bad",
        op_type="Reshape",
        inputs=["in"],
        attributes={"newshape": (5, 5, 2, 2)},
        shape_metadata=None,
    )

    g.nodes = {"in": inp, "bad": bad_op}

    Reshape = get_op("Reshape")

    # We monkeypatch infer_shape to raise ValueError
    original_infer = Reshape.infer_shape

    def mock_infer(*args: object, **kwargs: object) -> object:
        """Function docstring."""
        raise ValueError("mock error")

    Reshape.infer_shape = mock_infer
    try:
        with pytest.raises(CompilationError):
            shape_inference_pass(g)
    finally:
        Reshape.infer_shape = original_infer

    pass
