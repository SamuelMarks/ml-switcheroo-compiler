from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.autodiff_rules.shape_shape_rules import broadcast_to_jvp, broadcast_to_vjp, getitem_jvp, getitem_vjp, reshape_jvp, reshape_vjp, setitem_jvp, setitem_vjp, split_jvp, split_vjp, transpose_jvp, transpose_vjp
from ml_switcheroo_compiler.transforms.autodiff_rules.unary_math_rules import (
    abs_jvp,
    abs_vjp,
    acosh_jvp,
    acosh_vjp,
    cbrt_jvp,
    cbrt_vjp,
    cosh_jvp,
    cosh_vjp,
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


def test_shape_shape_rules():
    graph = LogicalGraph()
    inp_node = LogicalNode(id="inp1", op_type="Input", shape_metadata=(2, 3))
    graph.nodes["inp1"] = inp_node

    inp2_node = LogicalNode(id="inp2", op_type="Input", shape_metadata=(2, 3))
    graph.nodes["inp2"] = inp2_node

    # Reshape
    node = LogicalNode(id="n1", op_type="Reshape", inputs=["inp1"], attributes={"newshape": (6,)}, shape_metadata=(6,))
    assert reshape_jvp(graph, node, "t1")
    assert reshape_vjp(graph, node, "out_grad")

    # Transpose
    node.attributes = {"axes": [1, 0]}
    assert transpose_jvp(graph, node, "t1")
    assert transpose_vjp(graph, node, "out_grad")

    node.attributes = {}  # axes missing or None
    assert transpose_jvp(graph, node, "t1")
    assert transpose_vjp(graph, node, "out_grad")

    # BroadcastTo
    node.attributes = {"shape": (2, 2, 3)}
    assert broadcast_to_jvp(graph, node, "t1")
    assert broadcast_to_vjp(graph, node, "out_grad")

    # Split
    node.attributes = {"axis": 0, "indices_or_sections": 2}
    node.inputs = ["inp1"]
    assert split_jvp(graph, node, "t1")
    assert split_vjp(graph, node, ("out_grad1", "out_grad2"))

    # GetItem
    node.attributes = {"slices": "..."}
    assert getitem_jvp(graph, node, "t1")
    assert getitem_vjp(graph, node, "out_grad")

    node.attributes = {"key": "0"}
    assert getitem_jvp(graph, node, "t1")
    assert getitem_vjp(graph, node, "out_grad")

    # SetItem
    node.inputs = ["inp1", "inp2"]
    node.attributes = {"slices": "..."}
    assert setitem_jvp(graph, node, ("t1", "t2"))
    assert setitem_vjp(graph, node, "out_grad")

    node.attributes = {"key": "0"}
    assert setitem_jvp(graph, node, ("t1", "t2"))
    assert setitem_vjp(graph, node, "out_grad")


def test_unary_math_rules():
    graph = LogicalGraph()
    inp_node = LogicalNode(id="inp1", op_type="Input", shape_metadata=(2, 3))
    graph.nodes["inp1"] = inp_node

    node = LogicalNode(id="n1", op_type="Abs", inputs=["inp1"], shape_metadata=(2, 3))

    def run_vjp_jvp(vjp_func, jvp_func):
        vjp_res = vjp_func(graph, node, "out_grad")
        jvp_res = jvp_func(graph, node, ("t1",))
        assert vjp_res
        assert jvp_res

    funcs = [
        (abs_vjp, abs_jvp),
        (exp_vjp, exp_jvp),
        (exp2_vjp, exp2_jvp),
        (expm1_vjp, expm1_jvp),
        (log_vjp, log_jvp),
        (log10_vjp, log10_jvp),
        (log1p_vjp, log1p_jvp),
        (log2_vjp, log2_jvp),
        (sqrt_vjp, sqrt_jvp),
        (rsqrt_vjp, rsqrt_jvp),
        (square_vjp, square_jvp),
        (negative_vjp, negative_jvp),
        (positive_vjp, positive_jvp),
        (reciprocal_vjp, reciprocal_jvp),
        (cbrt_vjp, cbrt_jvp),
        (cosh_vjp, cosh_jvp),
        (acosh_vjp, acosh_jvp),
    ]

    for v, j in funcs:
        run_vjp_jvp(v, j)


def test_shape_shape_rules_edge_cases():
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.transforms.autodiff_rules.shape_shape_rules import setitem_jvp, split_vjp

    graph = LogicalGraph()
    inp_node = LogicalNode(id="inp1", op_type="Input", shape_metadata=(2, 3))
    graph.nodes["inp1"] = inp_node
    inp2_node = LogicalNode(id="inp2", op_type="Input", shape_metadata=(2, 3))
    graph.nodes["inp2"] = inp2_node

    # Split VJP with empty cotangents
    node = LogicalNode(id="n1", op_type="Split", inputs=["inp1"], shape_metadata=(2,))
    assert split_vjp(graph, node, ()) == ()

    # SetItem JVP with single tangent
    node = LogicalNode(id="n2", op_type="SetItem", inputs=["inp1", "inp2"], shape_metadata=(2, 3))
    assert setitem_jvp(graph, node, "t1")


def test_jvp_returns_none():
    from ml_switcheroo_ir import LogicalGraph
    from ml_switcheroo_ir import LogicalNode as Node

    from ml_switcheroo_compiler.transforms.autodiff import _forward_pass_jvp
    from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp

    @register_jvp("DummyReturnNoneForJVP")
    def dummy_none_jvp_rule(graph, node, tangents):
        return None

    graph = LogicalGraph()
    n1 = Node("n1", "Input")
    graph.nodes["n1"] = n1

    n2 = Node("n2", "DummyReturnNoneForJVP", inputs=["n1"])
    graph.nodes["n2"] = n2

    tangents = {"n1": "n1_tangent"}
    _forward_pass_jvp(graph, [n1, n2], tangents)
    assert "n2" not in tangents
