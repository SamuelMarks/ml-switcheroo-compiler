# ruff: noqa: E501
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.autodiff_rules.binary_math_rules import (
    add_jvp,
    add_vjp,
    divide_jvp,
    divide_no_nan_jvp,
    divide_no_nan_vjp,
    divide_vjp,
    fmax_jvp,
    fmax_vjp,
    fmin_jvp,
    fmin_vjp,
    hypot_jvp,
    hypot_vjp,
    logaddexp2_jvp,
    logaddexp2_vjp,
    logaddexp_jvp,
    logaddexp_vjp,
    maximum_jvp,
    maximum_vjp,
    minimum_jvp,
    minimum_vjp,
    multiply_jvp,
    multiply_no_nan_jvp,
    multiply_no_nan_vjp,
    multiply_vjp,
    power_jvp,
    power_vjp,
    squared_difference_jvp,
    squared_difference_vjp,
    subtract_jvp,
    subtract_vjp,
    xdivy_jvp,
    xdivy_vjp,
    xlog1py_jvp,
    xlog1py_vjp,
)


def create_mock_graph():
    graph = LogicalGraph()
    n1 = LogicalNode("x", "Input", shape_metadata=(2,))
    n2 = LogicalNode("y", "Input", shape_metadata=(2,))
    n3 = LogicalNode("n3", "Op", inputs=["x", "y"], shape_metadata=(2,))
    graph.nodes["x"] = n1
    graph.nodes["y"] = n2
    graph.nodes["n3"] = n3
    return (graph, n3)


def test_add(mocker):
    (graph, node) = create_mock_graph()
    assert add_vjp(graph, node, "cot") == ("cot", "cot")
    assert add_jvp("tx", "ty", "x", "y") == "(tx + ty)"


def test_subtract(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_math_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    graph.nodes["cot"] = LogicalNode("cot", "Input", shape_metadata=(2,))
    assert subtract_vjp(graph, node, "cot") == ("cot", "node")
    assert subtract_jvp("tx", "ty", "x", "y") == "(tx - ty)"


def test_multiply(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_math_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    graph.nodes["cot"] = LogicalNode("cot", "Input", shape_metadata=(2,))
    assert multiply_vjp(graph, node, "cot") == ("node", "node")
    assert multiply_jvp("tx", "ty", "x", "y") == "(tx * y + x * ty)"


def test_divide(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_math_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert divide_vjp(graph, node, "cot") == ("node", "node")
    assert divide_jvp("tx", "ty", "x", "y") == "((tx * y - x * ty) / (y * y))"


def test_power(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_math_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert power_vjp(graph, node, "cot") == ("node", "node")
    assert power_jvp("tx", "ty", "x", "y") == "(tx * y * x ** (y - 1) + ty * x ** y * log(x))"


def test_divide_no_nan(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_math_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert divide_no_nan_vjp(graph, node, "cot") == ("node", "node")
    assert divide_no_nan_jvp(graph, node, ("tx", "ty")) == "node"


def test_multiply_no_nan(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_math_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert multiply_no_nan_vjp(graph, node, "cot") == ("node", "node")
    assert multiply_no_nan_jvp(graph, node, ("tx", "ty")) == "node"


def test_squared_difference(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_math_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert squared_difference_vjp(graph, node, "cot") == ("node", "node")
    assert squared_difference_jvp(graph, node, ("tx", "ty")) == "node"


def test_xdivy(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_math_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert xdivy_vjp(graph, node, "cot") == ("node", "node")
    assert xdivy_jvp(graph, node, ("tx", "ty")) == "node"


def test_xlog1py(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_math_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert xlog1py_vjp(graph, node, "cot") == ("node", "node")
    assert xlog1py_jvp(graph, node, ("tx", "ty")) == "node"


def test_maximum_minimum(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_math_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert maximum_vjp(graph, node, "cot") == ("node", "node")
    assert maximum_jvp("tx", "ty", "x", "y", graph=graph) == "node"
    assert minimum_vjp(graph, node, "cot") == ("node", "node")
    assert minimum_jvp("tx", "ty", "x", "y", graph=graph) == "node"


def test_fmax_fmin(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_math_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert fmax_vjp(graph, node, "cot") == ("node", "node")
    assert fmax_jvp("tx", "ty", "x", "y", graph=graph) == "node"
    assert fmin_vjp(graph, node, "cot") == ("node", "node")
    assert fmin_jvp("tx", "ty", "x", "y", graph=graph) == "node"


def test_hypot(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_math_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert hypot_vjp(graph, node, "cot") == ("node", "node")
    assert hypot_jvp("tx", "ty", "x", "y", graph=graph) == "node"


def test_logaddexp(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.binary_math_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert logaddexp_vjp(graph, node, "cot") == ("node", "node")
    assert logaddexp_jvp("tx", "ty", "x", "y", graph=graph) == "node"
    assert logaddexp2_vjp(graph, node, "cot") == ("node", "node")
    assert logaddexp2_jvp("tx", "ty", "x", "y", graph=graph) == "node"
