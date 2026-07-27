# ruff: noqa: E501
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.autodiff_rules.binary_division_rules import floordivide_jvp, floordivide_vjp, fmod_jvp, fmod_vjp, remainder_jvp, remainder_vjp, truncatediv_jvp, truncatediv_vjp, truncatemod_jvp, truncatemod_vjp


def create_mock_graph():
    graph = LogicalGraph()
    n1 = LogicalNode("x", "Input", shape_metadata=(2,))
    n2 = LogicalNode("y", "Input", shape_metadata=(2,))
    n3 = LogicalNode("n3", "Op", inputs=["x", "y"], shape_metadata=(2,))
    graph.nodes["x"] = n1
    graph.nodes["y"] = n2
    graph.nodes["n3"] = n3
    return (graph, n3)


def test_truncatediv(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_division_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert truncatediv_vjp(graph, node, "cot") == ("node", "node")
    assert truncatediv_jvp("tx", "ty", "x", "y", graph=graph) == "node"


def test_truncatemod(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_division_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert truncatemod_vjp(graph, node, "cot") == ("cot", "node")
    assert truncatemod_jvp("tx", "ty", "x", "y", graph=graph) == "node"


def test_floordivide(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_division_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert floordivide_vjp(graph, node, "cot") == ("node", "node")
    assert floordivide_jvp("tx", "ty", "x", "y", graph=graph) == "node"


def test_remainder(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_division_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert remainder_vjp(graph, node, "cot") == ("cot", "node")
    assert remainder_jvp("tx", "ty", "x", "y", graph=graph) == "node"


def test_fmod(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_division_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert fmod_vjp(graph, node, "cot") == ("cot", "node")
    assert fmod_jvp("tx", "ty", "x", "y", graph=graph) == "node"
