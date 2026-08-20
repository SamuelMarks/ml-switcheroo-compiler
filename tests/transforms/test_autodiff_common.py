from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients, make_zero_jvp, make_zero_vjp


def test_make_zero_vjp():
    vjp_fn = make_zero_vjp("TestOp")
    graph = IRGraph()
    node = IRNode(id="test_node", op_type="TestOp", inputs=["a", "b"])
    res = vjp_fn(graph, node, "cot")
    assert len(res) == 2
    assert res[0] == UnconnectedGradients.ZERO
    assert res[1] == UnconnectedGradients.ZERO


def test_make_zero_jvp():
    jvp_fn = make_zero_jvp("TestOp")
    graph = IRGraph()
    node = IRNode(id="test_node", op_type="TestOp", inputs=["a", "b"])
    res = jvp_fn(graph, node, ("t_a", "t_b"))
    assert res == ""
