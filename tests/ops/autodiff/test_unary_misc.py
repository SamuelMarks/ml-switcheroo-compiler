# ruff: noqa: E501
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.autodiff_rules.unary_misc_rules import _zero_jvp, _zero_vjp, cast_jvp, cast_vjp, conj_jvp, conj_vjp


def create_mock_graph():
    graph = LogicalGraph()
    n1 = LogicalNode("x", "Input", shape_metadata=(2,))
    n2 = LogicalNode("n2", "Op", inputs=["x"], shape_metadata=(2,), attributes={"dtype": "float32"})
    graph.nodes["x"] = n1
    graph.nodes["n2"] = n2
    return (graph, n2)


def test_unary_misc_rules(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.unary_misc_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert _zero_vjp(graph, node, "cot") == ("node",)
    assert _zero_jvp(graph, node, ("t",)) == "node"
    assert conj_vjp(graph, node, "cot") == ("node",)
    assert conj_jvp(graph, node, ("t",)) == "node"
    assert cast_vjp(graph, node, "cot") == ("node",)
    assert cast_jvp(graph, node, ("t",)) == "node"
