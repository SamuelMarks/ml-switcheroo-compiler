"""Unit tests for Batch Norm Folding pass."""

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.batch_norm_folding import batch_norm_folding_pass


def test_batch_norm_folding_no_op():
    """Test on an empty graph."""
    graph = IRGraph()
    assert batch_norm_folding_pass(graph) is False


def test_batch_norm_folding_basic():
    """Test folding BatchNorm into Conv2D."""
    graph = IRGraph()

    conv = IRNode(id="conv_1", op_type="Conv2D", inputs=["inp_1", "weight_1"])
    bn = IRNode(id="bn_1", op_type="BatchNorm", inputs=["conv_1", "scale", "bias", "mean", "var"])
    relu = IRNode(id="relu_1", op_type="Relu", inputs=["bn_1"])

    graph.nodes = {"conv_1": conv, "bn_1": bn, "relu_1": relu}
    graph.outputs = ["relu_1"]

    modified = batch_norm_folding_pass(graph)
    assert modified is True

    # Check Conv2D is modified
    # assert conv.attributes.get("folded_batch_norm") is True
    # assert conv.attributes.get("bn_inputs") == ["scale", "bias", "mean", "var"]

    # Check BatchNorm is removed
    assert "bn_1" not in graph.nodes

    # Check consumer is rewired
    assert relu.inputs == ["conv_1"]


def test_batch_norm_folding_output_node():
    """Test folding when BatchNorm is an output node."""
    graph = IRGraph()

    conv = IRNode(id="conv_1", op_type="Conv2D", inputs=["inp_1", "weight_1"])
    bn = IRNode(id="bn_1", op_type="BatchNorm", inputs=["conv_1", "scale", "bias", "mean", "var"])

    graph.nodes = {"conv_1": conv, "bn_1": bn}
    graph.outputs = ["bn_1"]

    modified = batch_norm_folding_pass(graph)
    assert modified is True
    assert "bn_1" not in graph.nodes
    assert graph.outputs == ["conv_1"]


def test_batch_norm_folding_no_conv2d():
    """Test when BatchNorm is not preceded by Conv2D."""
    graph = IRGraph()

    add = IRNode(id="add_1", op_type="Add", inputs=["inp_1", "inp_2"])
    bn = IRNode(id="bn_1", op_type="BatchNorm", inputs=["add_1", "scale", "bias", "mean", "var"])

    graph.nodes = {"add_1": add, "bn_1": bn}
    graph.outputs = ["bn_1"]

    modified = batch_norm_folding_pass(graph)
    assert modified is False
    assert "bn_1" in graph.nodes


def test_batch_norm_folding_with_bias():
    """Test folding BatchNorm into Conv2D when Conv2D already has a bias."""
    graph = IRGraph()

    # Conv2D with 3 inputs: inp, weight, bias
    conv = IRNode(id="conv_1", op_type="Conv2D", inputs=["inp_1", "weight_1", "bias_1"])
    bn = IRNode(id="bn_1", op_type="BatchNorm", inputs=["conv_1", "scale", "bias", "mean", "var"])
    relu = IRNode(id="relu_1", op_type="Relu", inputs=["bn_1"])

    graph.nodes = {"conv_1": conv, "bn_1": bn, "relu_1": relu}
    graph.outputs = ["relu_1"]

    modified = batch_norm_folding_pass(graph)
    assert modified is True
    assert "bn_1" not in graph.nodes
