# ruff: noqa: E501
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.autodiff_rules.binary_trig_rules import atan2_jvp, atan2_vjp


def create_mock_graph():
    graph = LogicalGraph()
    n1 = LogicalNode("y", "Input", shape_metadata=(2,))
    n2 = LogicalNode("x", "Input", shape_metadata=(2,))
    n3 = LogicalNode("n3", "Op", inputs=["y", "x"], shape_metadata=(2,))
    graph.nodes["y"] = n1
    graph.nodes["x"] = n2
    graph.nodes["n3"] = n3
    return (graph, n3)


def test_atan2(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_trig_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert atan2_vjp(graph, node, "cot") == ("node", "node")
    assert atan2_jvp("ty", "tx", "y", "x", graph=graph) == "node"
