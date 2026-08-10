"""Unit tests for Axis Translation pass."""

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.axis_translation import axis_translation_pass


def test_axis_translation_no_op():
    """Test axis translation on an empty graph."""
    graph = IRGraph()
    assert axis_translation_pass(graph) is False


def test_axis_translation_conv2d():
    """Test inserting Transpose nodes around Conv2D."""
    graph = IRGraph()

    # Input node
    shape_meta = (1, 3, 32, 32)
    inp = IRNode(id="inp_1", op_type="Input", shape_metadata=shape_meta)

    # Conv2D node (default layout is NCHW)
    conv = IRNode(id="conv_1", op_type="Conv2D", inputs=["inp_1"], attributes={"layout": "NCHW"}, shape_metadata=shape_meta)

    # Consumer node
    relu = IRNode(id="relu_1", op_type="Relu", inputs=["conv_1"], shape_metadata=shape_meta)

    graph.nodes = {"inp_1": inp, "conv_1": conv, "relu_1": relu}
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


def test_axis_translation_conv2d_no_shape():
    """Test inserting Transpose nodes around Conv2D when shape is None."""
    graph = IRGraph()
    inp = IRNode(id="inp_1", op_type="Input", shape_metadata=None)
    conv = IRNode(id="conv_1", op_type="Conv2D", inputs=["inp_1"], attributes={}, shape_metadata=None)
    relu = IRNode(id="relu_1", op_type="Relu", inputs=["conv_1"], shape_metadata=None)

    # We output both `relu` and `conv_1` to hit line 62
    graph.nodes = {"inp_1": inp, "conv_1": conv, "relu_1": relu}
    graph.outputs = ["relu_1", "conv_1"]

    # We manually add `transp_out_id` from the future to the graph to hit the `continue` at line 57
    import uuid

    dummy_transp = IRNode(id="transpose_dummy", op_type="Transpose")
    graph.nodes["transpose_dummy"] = dummy_transp

    # Monkey patch uuid to force the dummy name
    original_uuid4 = uuid.uuid4

    def mock_uuid4():
        class MockUUID:
            hex = "dummyxxx"

        return MockUUID()

    uuid.uuid4 = mock_uuid4
    try:
        modified = axis_translation_pass(graph)
    finally:
        uuid.uuid4 = original_uuid4

    assert modified is True
    assert conv.attributes["layout"] == "NHWC"
    assert graph.nodes[conv.inputs[0]].op_type == "Transpose"


def test_axis_translation_conv2d_already_nhwc():
    """Test axis translation skips Conv2D if already NHWC."""
    graph = IRGraph()
    conv = IRNode(id="conv_1", op_type="Conv2D", inputs=[], attributes={"layout": "NHWC"})
    graph.nodes = {"conv_1": conv}
    modified = axis_translation_pass(graph)
    assert modified is False
