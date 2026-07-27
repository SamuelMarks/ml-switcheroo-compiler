# ruff: noqa: E501
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.autodiff_rules.unary_math_rules import (
    abs_jvp,
    abs_vjp,
    cbrt_jvp,
    cbrt_vjp,
    exp2_jvp,
    exp2_vjp,
    exp_jvp,
    exp_vjp,
    expm1_jvp,
    expm1_vjp,
    log1p_jvp,
    log1p_vjp,
    log2_jvp,
    log2_vjp,
    log10_jvp,
    log10_vjp,
    log_jvp,
    log_vjp,
    negative_jvp,
    negative_vjp,
    positive_jvp,
    positive_vjp,
    reciprocal_jvp,
    reciprocal_vjp,
    rsqrt_jvp,
    rsqrt_vjp,
    sqrt_jvp,
    sqrt_vjp,
    square_jvp,
    square_vjp,
)


def create_mock_graph():
    graph = LogicalGraph()
    n1 = LogicalNode("x", "Input", shape_metadata=(2,))
    n2 = LogicalNode("n2", "Op", inputs=["x"], shape_metadata=(2,))
    graph.nodes["x"] = n1
    graph.nodes["n2"] = n2
    return (graph, n2)


def test_unary_math_vjp(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.unary_math_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert abs_vjp(graph, node, "cot") == ("node",)
    assert exp_vjp(graph, node, "cot") == ("node",)
    assert exp2_vjp(graph, node, "cot") == ("node",)
    assert expm1_vjp(graph, node, "cot") == ("node",)
    assert log_vjp(graph, node, "cot") == ("node",)
    assert log10_vjp(graph, node, "cot") == ("node",)
    assert log1p_vjp(graph, node, "cot") == ("node",)
    assert log2_vjp(graph, node, "cot") == ("node",)
    assert sqrt_vjp(graph, node, "cot") == ("node",)
    assert rsqrt_vjp(graph, node, "cot") == ("node",)
    assert square_vjp(graph, node, "cot") == ("node",)
    assert negative_vjp(graph, node, "cot") == ("node",)
    assert positive_vjp(graph, node, "cot") == ("node",)
    assert reciprocal_vjp(graph, node, "cot") == ("node",)
    assert cbrt_vjp(graph, node, "cot") == ("node",)


def test_unary_math_jvp(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.unary_math_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert abs_jvp(graph, node, ("t",)) == "node"
    assert exp_jvp(graph, node, ("t",)) == "node"
    assert exp2_jvp(graph, node, ("t",)) == "node"
    assert expm1_jvp(graph, node, ("t",)) == "node"
    assert log_jvp(graph, node, ("t",)) == "node"
    assert log10_jvp(graph, node, ("t",)) == "node"
    assert log1p_jvp(graph, node, ("t",)) == "node"
    assert log2_jvp(graph, node, ("t",)) == "node"
    assert sqrt_jvp(graph, node, ("t",)) == "node"
    assert rsqrt_jvp(graph, node, ("t",)) == "node"
    assert square_jvp(graph, node, ("t",)) == "node"
    assert negative_jvp(graph, node, ("t",)) == "node"
    assert positive_jvp(graph, node, ("t",)) == "node"
    assert reciprocal_jvp(graph, node, ("t",)) == "node"
    assert cbrt_jvp(graph, node, ("t",)) == "node"
