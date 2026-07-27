# ruff: noqa: E501
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.autodiff_rules.shape_misc_rules import _generic_shape_jvp, _generic_shape_vjp


def test_shape_misc_rules(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.shape_misc_rules.emit_ir_node", return_value="node")
    graph = LogicalGraph()
    n1 = LogicalNode("x", "Input", shape_metadata=(2,))
    n2 = LogicalNode("n2", "Op", inputs=["x"], shape_metadata=(2, 2))
    graph.nodes["x"] = n1
    graph.nodes["n2"] = n2
    assert _generic_shape_vjp(graph, n2, "cot") == ("node",)
    assert _generic_shape_jvp(graph, n2, "t1") == "node"
