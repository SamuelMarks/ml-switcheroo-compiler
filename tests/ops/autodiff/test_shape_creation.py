# ruff: noqa: E501
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients
from ml_switcheroo_compiler.transforms.autodiff_rules.shape_creation_rules import _creation_jvp, _creation_vjp, full_jvp, full_vjp


def test_creation_rules(mocker):
    graph = LogicalGraph()
    n1 = LogicalNode("n1", "Input", shape_metadata=(2,))
    n2 = LogicalNode("n2", "Input", shape_metadata=(2,))
    n_op = LogicalNode("n_op", "Zeros", inputs=["n1", "n2"], shape_metadata=(2,))
    graph.nodes["n1"] = n1
    graph.nodes["n2"] = n2
    graph.nodes["n_op"] = n_op
    assert _creation_vjp(graph, n_op, "cot") == (UnconnectedGradients.ZERO, UnconnectedGradients.ZERO)
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.shape_creation_rules.emit_ir_node", return_value="node")
    assert _creation_jvp(graph, n_op, ("t1", "t2")) == "node"


def test_full_rules(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.shape_creation_rules.emit_ir_node", return_value="node")
    graph = LogicalGraph()
    n1 = LogicalNode("shape", "Input", shape_metadata=(2,))
    n2 = LogicalNode("fill_value", "Input", shape_metadata=())
    n_op = LogicalNode("n_op", "Full", inputs=["shape", "fill_value"], shape_metadata=(2, 2))
    graph.nodes["shape"] = n1
    graph.nodes["fill_value"] = n2
    graph.nodes["n_op"] = n_op
    assert full_vjp(graph, n_op, "cot") == (UnconnectedGradients.ZERO, "node")
    assert full_jvp(graph, n_op, ("t1", "t2")) == "node"
    n_op2 = LogicalNode("n_op2", "Full", inputs=["shape"], shape_metadata=(2, 2))
    graph.nodes["n_op2"] = n_op2
    assert full_vjp(graph, n_op2, "cot") == (UnconnectedGradients.ZERO,)
    assert full_jvp(graph, n_op2, ("t1",)) == "node"
