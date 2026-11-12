"""Unit tests for Axis Translation pass."""

from __future__ import annotations

import typing

from ml_switcheroo_ir import NodeDict

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.axis_translation import axis_translation_pass
from tests.testing_utils import ensure_eager_tensor, ensure_mutable, has_backend_member

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_axis_translation_no_op() -> None:
    """Test axis translation on an empty graph."""
    graph = IRGraph()
    assert axis_translation_pass(graph) is False


def test_axis_translation_conv2d() -> None:
    """Test inserting Transpose nodes around Conv2D."""
    graph = IRGraph()

    # Input node
    shape_meta = (1, 3, 32, 32)
    inp = IRNode(id="inp_1", op_type="Input", shape_metadata=shape_meta)

    # Conv2D node (default layout is NCHW)
    conv = IRNode(id="conv_1", op_type="Conv2D", inputs=["inp_1"], attributes={"layout": "NCHW"}, shape_metadata=shape_meta)

    # Consumer node
    relu = IRNode(id="relu_1", op_type="Relu", inputs=["conv_1"], shape_metadata=shape_meta)

    graph.nodes = NodeDict({"inp_1": inp, "conv_1": conv, "relu_1": relu})
    graph.outputs = ["relu_1"]

    modified = axis_translation_pass(graph)
    assert modified is True

    # Check that Conv2D layout was changed to NHWC
    assert conv.attributes["layout"] == "NHWC"

    # Check that Transpose nodes were inserted
    # 1. Before Conv2D
    transp_in_id = conv.inputs[0]
    assert transp_in_id.startswith("transpose_")
    transp_in_node = graph.nodes[transp_in_id]
    assert transp_in_node.op_type == "Transpose"
    assert transp_in_node.inputs == ["inp_1"]
    assert transp_in_node.attributes["axes"] == [0, 2, 3, 1]

    # 2. After Conv2D (Relu input)
    transp_out_id = relu.inputs[0]
    assert transp_out_id.startswith("transpose_")
    assert transp_out_id != transp_in_id
    transp_out_node = graph.nodes[transp_out_id]
    assert transp_out_node.op_type == "Transpose"
    assert transp_out_node.inputs == ["conv_1"]
    assert transp_out_node.attributes["axes"] == [0, 3, 1, 2]

    # 3. Test with 2 inputs (input + 4D weight and unshaped weight)
    g2 = IRGraph()
    g2.nodes = NodeDict(
        {
            "x": IRNode(id="x", op_type="Input", shape_metadata=(1, 3, 8, 8)),
            "w": IRNode(id="w", op_type="Input", shape_metadata=(16, 3, 3, 3)),
            "conv": IRNode(id="conv", op_type="Conv2D", inputs=["x", "w"], attributes={"layout": "NCHW"}, shape_metadata=(1, 16, 6, 6)),
        }
    )
    g2.outputs = ["conv"]
    assert axis_translation_pass(g2) is True
    assert g2.nodes["conv"].attributes["kernel_layout"] == "HWIO"
    w_transp_id = g2.nodes["conv"].inputs[1]
    assert g2.nodes[w_transp_id].shape_metadata == (3, 3, 3, 16)
    assert g2.nodes[w_transp_id].attributes["axes"] == [2, 3, 1, 0]

    g3 = IRGraph()
    g3.nodes = NodeDict(
        {
            "x": IRNode(id="x", op_type="Input"),
            "w": IRNode(id="w", op_type="Input"),
            "conv": IRNode(id="conv", op_type="Conv2D", inputs=["x", "w"], attributes={"layout": "NCHW"}),
        }
    )
    assert axis_translation_pass(g3) is True
    assert g3.nodes[g3.nodes["conv"].inputs[1]].shape_metadata is None


def test_axis_translation_conv2d_no_shape() -> None:
    """Test inserting Transpose nodes around Conv2D when shape is None."""
    graph = IRGraph()
    inp = IRNode(id="inp_1", op_type="Input", shape_metadata=None)
    conv = IRNode(id="conv_1", op_type="Conv2D", inputs=["inp_1"], attributes={}, shape_metadata=None)
    relu = IRNode(id="relu_1", op_type="Relu", inputs=["conv_1"], shape_metadata=None)

    # We output both `relu` and `conv_1` to hit line 62
    graph.nodes = NodeDict({"inp_1": inp, "conv_1": conv, "relu_1": relu})
    graph.outputs = ["relu_1", "conv_1"]

    # We manually add `transp_out_id` from the future to the graph to hit the `continue` at line 57
    import uuid

    dummy_transp = IRNode(id="transpose_dummy", op_type="Transpose")
    graph.nodes["transpose_dummy"] = dummy_transp

    # Monkey patch uuid to force the dummy name
    original_uuid4 = uuid.uuid4

    def mock_uuid4() -> EagerValue:
        """Test function mock_uuid4.

        Returns:
            EagerValue: Computed output value from `mock_uuid4`.
        """

        class MockUUID:
            """Test class MockUUID."""

            hex = "dummyxxx"

        return ensure_eager_tensor(MockUUID())

    ensure_mutable(uuid).uuid4 = ensure_eager_tensor(mock_uuid4)
    try:
        modified = axis_translation_pass(graph)
    finally:
        uuid.uuid4 = original_uuid4

    assert modified is True
    assert conv.attributes["layout"] == "NHWC"
    assert graph.nodes[conv.inputs[0]].op_type == "Transpose"


def test_axis_translation_conv2d_already_nhwc() -> None:
    """Test axis translation skips Conv2D if already NHWC."""
    graph = IRGraph()
    conv = IRNode(id="conv_1", op_type="Conv2D", inputs=[], attributes={"layout": "NHWC"})
    graph.nodes = NodeDict({"conv_1": conv})
    modified = axis_translation_pass(graph)
    assert modified is False


def test_axis_translation_conv2d_empty_inputs() -> None:
    """Test Conv2D with layout NCHW but empty inputs."""
    graph = IRGraph()
    conv = IRNode(id="conv_1", op_type="Conv2D", inputs=[], attributes={"layout": "NCHW"})
    graph.nodes = NodeDict({"conv_1": conv})
    assert axis_translation_pass(graph) is True


def test_passes_package_init() -> None:
    """Test passes package __init__ exports."""
    import ml_switcheroo_compiler.transforms.passes as passes

    assert has_backend_member(passes, "axis_translation_pass")
    assert passes.axis_translation_pass is not None
