# ruff: noqa: E501
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.autodiff_rules.binary_special_rules import UnconnectedGradients, betainc_jvp, betainc_vjp, igamma_jvp, igamma_vjp, igammac_jvp, igammac_vjp, polygamma_jvp, polygamma_vjp, zeta_jvp, zeta_vjp


def create_mock_graph_2_inputs():
    graph = LogicalGraph()
    n1 = LogicalNode("x", "Input", shape_metadata=(2,))
    n2 = LogicalNode("y", "Input", shape_metadata=(2,))
    n3 = LogicalNode("n3", "Op", inputs=["x", "y"], shape_metadata=(2,))
    graph.nodes["x"] = n1
    graph.nodes["y"] = n2
    graph.nodes["n3"] = n3
    return (graph, n3)


def create_mock_graph_3_inputs():
    graph = LogicalGraph()
    n1 = LogicalNode("a", "Input", shape_metadata=(2,))
    n2 = LogicalNode("b", "Input", shape_metadata=(2,))
    n3 = LogicalNode("x", "Input", shape_metadata=(2,))
    n4 = LogicalNode("n4", "Op", inputs=["a", "b", "x"], shape_metadata=(2,))
    graph.nodes["a"] = n1
    graph.nodes["b"] = n2
    graph.nodes["x"] = n3
    graph.nodes["n4"] = n4
    return (graph, n4)


def test_igamma(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_special_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph_2_inputs()
    assert igamma_vjp(graph, node, "cot") == ("node", "node")
    assert igamma_jvp(graph, node, ("tx", "ty")) == "node"


def test_igammac(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_special_rules.emit_ir_node", return_value="node")
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_special_rules.igamma_vjp", return_value=("da", "dx"))
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_special_rules.igamma_jvp", return_value="node")
    (graph, node) = create_mock_graph_2_inputs()
    assert igammac_vjp(graph, node, "cot") == ("node", "node")
    assert igammac_jvp(graph, node, ("tx", "ty")) == "node"


def test_zeta(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_special_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph_2_inputs()
    assert zeta_vjp(graph, node, "cot") == (UnconnectedGradients.ZERO, "node")
    assert zeta_jvp(graph, node, ("tx", "ty")) == "node"
    assert zeta_jvp(graph, node, ("tx", None)) == ""


def test_polygamma(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_special_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph_2_inputs()
    assert polygamma_vjp(graph, node, "cot") == (UnconnectedGradients.ZERO, "node")
    assert polygamma_jvp(graph, node, ("tx", "ty")) == "node"


def test_betainc(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_special_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph_3_inputs()
    assert betainc_vjp(graph, node, "cot") == (UnconnectedGradients.ZERO, UnconnectedGradients.ZERO, "node")
    assert betainc_jvp(graph, node, ("ta", "tb", "tx")) == "node"
