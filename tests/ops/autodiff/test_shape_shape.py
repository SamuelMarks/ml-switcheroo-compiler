# ruff: noqa: E501
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.autodiff_rules.shape_shape_rules import broadcast_to_jvp, broadcast_to_vjp, reshape_jvp, reshape_vjp, split_jvp, split_vjp, transpose_jvp, transpose_vjp


def create_mock_graph():
    graph = LogicalGraph()
    n1 = LogicalNode("x", "Input", shape_metadata=(2,))
    n3 = LogicalNode("n3", "Op", inputs=["x"], shape_metadata=(2, 2))
    graph.nodes["x"] = n1
    graph.nodes["n3"] = n3
    return (graph, n3)


def test_reshape(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.shape_shape_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    node.attributes = {"newshape": (2, 2)}
    assert reshape_vjp(graph, node, "cot") == ("node",)
    assert reshape_jvp(graph, node, "t1") == "jnp.reshape(t1, (2, 2))"


def test_transpose(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.shape_shape_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    node.attributes = {}
    assert transpose_vjp(graph, node, "cot") == ("node",)
    assert transpose_jvp(graph, node, "t1") == "jnp.transpose(t1)"
    node.attributes = {"axes": [1, 0]}
    assert transpose_vjp(graph, node, "cot") == ("node",)
    assert transpose_jvp(graph, node, "t1") == "jnp.transpose(t1, axes=[1, 0])"


def test_broadcast_to(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.shape_shape_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    node.attributes = {"shape": (2, 2)}
    assert broadcast_to_vjp(graph, node, "cot") == ("node",)
    assert broadcast_to_jvp(graph, node, "t1") == "jnp.broadcast_to(t1, (2, 2))"


def test_split(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.shape_shape_rules.emit_ir_node", return_value="node")
    (graph, node) = create_mock_graph()
    assert split_vjp(graph, node, ()) == ()
    res = split_vjp(graph, node, ("cot1", "cot2"))
    assert len(res) == 1
    assert split_jvp(graph, node, ("t1",)) is not None
