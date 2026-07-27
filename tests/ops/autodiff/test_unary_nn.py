# ruff: noqa: E501
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.autodiff_rules.unary_nn_rules import logit_jvp, logit_vjp, logsigmoid_jvp, logsigmoid_vjp, sigmoid_jvp, sigmoid_vjp


def create_mock_graph():
    graph = LogicalGraph()
    n1 = LogicalNode("x", "Input", shape_metadata=(2,))
    n2 = LogicalNode("n2", "Op", inputs=["x"], shape_metadata=(2,))
    graph.nodes["x"] = n1
    graph.nodes["n2"] = n2
    return (graph, n2)


def test_sigmoid(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.unary_nn_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert sigmoid_vjp(graph, node, "cot") == ("node",)
    assert sigmoid_jvp(graph, node, ("t1",)) == "node"


def test_logsigmoid(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.unary_nn_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert logsigmoid_vjp(graph, node, "cot") == ("node",)
    assert logsigmoid_jvp(graph, node, ("t1",)) == "node"


def test_logit(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.unary_nn_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert logit_vjp(graph, node, "cot") == ("node",)
    assert logit_jvp(graph, node, ("t1",)) == "node"
