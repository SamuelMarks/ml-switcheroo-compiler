# ruff: noqa: E501
import pytest

from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.shape_inference import _determine_node_shape, _infer_constant_shape, _infer_op_shape, _infer_output_shape, _prepare_op_kwargs, shape_inference_pass


def test_infer_constant_shape() -> None:
    node = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": [1.0, 2.0]})
    assert _infer_constant_shape(node, {}) == (2,)


def test_infer_output_shape() -> None:
    node1 = IRNode(id="n1", op_type="Output", inputs=["in1"], attributes={})
    assert _infer_output_shape(node1, {"in1": (3, 4)}) == (3, 4)
    node2 = IRNode(id="n2", op_type="Output", inputs=[], attributes={})
    assert _infer_output_shape(node2, {}) is None


def test_prepare_op_kwargs() -> None:
    node1 = IRNode(id="n1", op_type="Expand", inputs=[], attributes={"other": 1}, shape_metadata=(1, 2))
    assert _prepare_op_kwargs(node1) == {"other": 1, "shape": (1, 2)}
    node2 = IRNode(id="n2", op_type="BroadcastTo", inputs=[], attributes={}, shape_metadata=(3, 3))
    assert _prepare_op_kwargs(node2) == {"shape": (3, 3)}
    node3 = IRNode(id="n3", op_type="Reshape", inputs=[], attributes={}, shape_metadata=(4,))
    assert _prepare_op_kwargs(node3) == {"newshape": (4,)}
    node4 = IRNode(id="n4", op_type="Add", inputs=[], attributes={"other": 1}, shape_metadata=(1,))
    assert _prepare_op_kwargs(node4) == {"other": 1}


def test_infer_op_shape() -> None:
    node1 = IRNode(id="n1", op_type="Add", inputs=["in1", "in2"], attributes={})
    shapes = {"in1": (1, 2), "in2": (1, 2)}
    assert _infer_op_shape(node1, shapes) == (1, 2)


def test_determine_node_shape() -> None:
    node1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=(5, 5))
    assert _determine_node_shape(node1, {}) == (5, 5)
    node2 = IRNode(id="n2", op_type="UnknownOp", inputs=[], attributes={}, shape_metadata=(1,))
    assert _determine_node_shape(node2, {}) == (1,)
    node3 = IRNode(id="n3", op_type="Add", inputs=["in1", "in2"], attributes={})
    with pytest.raises(CompilationError):
        _determine_node_shape(node3, {"in1": (1, 2), "in2": (3, 4)})


def test_shape_inference_pass() -> None:
    node1 = IRNode(id="in1", op_type="Input", inputs=[], attributes={}, shape_metadata=(2, 2))
    node2 = IRNode(id="in2", op_type="Input", inputs=[], attributes={}, shape_metadata=(2, 2))
    node3 = IRNode(id="add", op_type="Add", inputs=["in1", "in2"], attributes={}, shape_metadata=None)
    graph = IRGraph(name="test", nodes={"in1": node1, "in2": node2, "add": node3}, outputs=["add"])
    assert shape_inference_pass(graph) is True
    assert graph.nodes["add"].shape_metadata == (2, 2)
