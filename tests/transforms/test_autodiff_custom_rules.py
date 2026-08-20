"""Coverage tests for custom rules autodiff."""

import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import (
    _assoc_scan_jvp,
    _assoc_scan_vjp,
    _if_jvp,
    _if_vjp,
    _inline_grad_subgraph,
    _inline_subgraph,
    _loop_jvp,
    _loop_vjp,
    _scan_jvp,
    _scan_vjp,
)


def test_inline_subgraph():
    g = LogicalGraph()
    sg = LogicalGraph()
    n = LogicalNode(id="n", op_type="Custom")
    sg.nodes["i"] = LogicalNode(id="i", op_type="Input")
    sg.nodes["a"] = LogicalNode(id="a", op_type="Add", inputs=["i"])
    _inline_subgraph(g, sg, n, {"i": "x", "a": "new_a"})
    assert "new_a" in g.nodes


def test_inline_grad_subgraph():
    g = LogicalGraph()
    sg = LogicalGraph()
    sg.inputs = ["i"]
    sg_grad = LogicalGraph()
    sg_grad.nodes["a"] = LogicalNode(id="a", op_type="Add", inputs=["i"])
    sg_grad.outputs = ["a"]
    n = LogicalNode(id="n", op_type="Custom", inputs=["i"])
    res = _inline_grad_subgraph(g, sg_grad, sg, n, {})
    assert len(res) == 1


def test_if_vjp():
    g = LogicalGraph()
    n = LogicalNode(id="n", op_type="If", inputs=["cond", "a", "b"], attributes={"true_branch": LogicalGraph(), "false_branch": LogicalGraph()})
    res = _if_vjp(g, n, "cot")
    assert len(res) >= 1


def test_loop_vjp():
    g = LogicalGraph()
    n = LogicalNode(id="n", op_type="Loop", inputs=["init", "n_iter"], attributes={"body": LogicalGraph()})
    res = _loop_vjp(g, n, "cot")
    assert len(res) >= 1


def test_scan_vjp():
    g = LogicalGraph()
    n = LogicalNode(id="n", op_type="Scan", inputs=["a", "b"], attributes={"body": LogicalGraph()})
    res = _scan_vjp(g, n, "cot")
    assert len(res) >= 1


def test_assoc_scan_vjp():
    g = LogicalGraph()
    n = LogicalNode(id="n", op_type="AssociativeScan", inputs=["a", "b", "c"], attributes={"combine_fn": LogicalGraph()})
    res = _assoc_scan_vjp(g, n, "cot")
    assert len(res) >= 1


def test_jvps():
    g = LogicalGraph()
    n1 = LogicalNode(id="n1", op_type="If", inputs=["cond", "a", "b"], attributes={"then_branch": LogicalGraph(), "else_branch": LogicalGraph()})
    assert _if_jvp(g, n1, ["t1", "t2", "t3"]) == "n1_jvp"
    with pytest.raises(ValueError):
        _if_jvp(g, n1, ["t1", "t2"])

    n2 = LogicalNode(id="n2", op_type="Loop", inputs=["init", "n_iter"], attributes={"body": LogicalGraph()})
    assert _loop_jvp(g, n2, ["t1", "t2"]) == ""
    assert _loop_jvp(g, n2, ["t1"]) == ""

    n3 = LogicalNode(id="n3", op_type="Scan", inputs=["a", "b"], attributes={"body": LogicalGraph()})
    assert _scan_jvp(g, n3, ["t1", "t2"]) == ""
    assert _scan_jvp(g, n3, ["t1"]) == ""

    n4 = LogicalNode(id="n4", op_type="AssociativeScan", inputs=["a", "b", "c"], attributes={"combine_fn": LogicalGraph()})
    assert _assoc_scan_jvp(g, n4, ["t1", "t2", "t3"]) == ""
    assert _assoc_scan_jvp(g, n4, ["t1", "t2"]) == ""


def test_jvp_nulls():
    pass


def test_checkpoint_vjp():
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import checkpoint_vjp

    # create a simple subgraph
    sg = IRGraph()
    sg.nodes["inp"] = IRNode(id="inp", op_type="Input")
    sg.nodes["out"] = IRNode(id="out", op_type="Exp", inputs=["inp"])
    sg.inputs = ["inp"]
    sg.outputs = ["out"]

    node = IRNode(id="cp", op_type="Checkpoint", inputs=["in_main"], attributes={"subgraph": sg})

    main_graph = IRGraph()
    main_graph.nodes["in_main"] = IRNode(id="in_main", op_type="Input")

    # We must patch get_vjp because Exp VJP is in another registry or we just mock graph_grad
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.transforms.autodiff.grad") as mock_grad:
        sg_grad = IRGraph()
        sg_grad.nodes["out"] = IRNode(id="out", op_type="Input")
        sg_grad.nodes["inp"] = IRNode(id="inp", op_type="Output", inputs=["out"])
        # Add an intermediate node that isn't Input/Output or cotangent mapping
        sg_grad.nodes["mid"] = IRNode(id="mid", op_type="Exp", inputs=["out"])
        # Add a node that is the cotangent to trigger continue
        sg_grad.nodes["cotangent_id"] = IRNode(id="cotangent_id", op_type="Exp")
        sg_grad.inputs = ["out"]
        sg_grad.outputs = ["inp"]
        mock_grad.return_value = sg_grad

        res = checkpoint_vjp(main_graph, node, "cotangent_id")
        assert res is not None


def test_if_jvp():
    from unittest.mock import patch

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import _assoc_scan_jvp, _if_jvp, _loop_jvp, _scan_jvp, recompute_vjp

    assert _scan_jvp(None, None, None) == ""
    assert _assoc_scan_jvp(None, None, None) == ""
    assert _loop_jvp(None, None, None) == ""

    n_recompute = IRNode(id="r", op_type="Recompute", inputs=["x"], attributes={"original_op": "Exp"})
    g_recompute = IRGraph()
    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry.get_vjp") as mock_get_vjp:
        mock_get_vjp.return_value = lambda g, n, cot: ("mocked_res",)
        assert recompute_vjp(g_recompute, n_recompute, "cot") == ("mocked_res",)

    with patch("ml_switcheroo_compiler.transforms.autodiff.jvp") as mock_jvp:
        mock_jvp.return_value = IRGraph()  # dummy

        g = IRGraph()
        n = IRNode(id="n1", op_type="If", inputs=["c", "x"])
        n.attributes = {"then_branch": IRGraph(), "else_branch": IRGraph()}

        res = _if_jvp(g, n, ["t_c", "t_x"])
        assert res == "n1_jvp"
        assert "n1_jvp" in g.nodes

        n_empty = IRNode(id="n2", op_type="If", inputs=["c"])
        res_empty = _if_jvp(g, n_empty, ["t_c"])
        assert res_empty == "mock_tangent"


def test_custom_vjp_vjp():
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import custom_vjp_vjp

    graph = IRGraph()
    node = IRNode(id="test_node", op_type="CustomVJP", inputs=["a", "b"], attributes={"bwd_fn": "fake_bwd_fn"})
    node.shape_metadata = ()

    res = custom_vjp_vjp(graph, node, "cot")
    assert len(res) == 2
    assert "ProcessCustomVJPCall" in [n.op_type for n in graph.nodes.values()]
    assert "TupleGetItem" in [n.op_type for n in graph.nodes.values()]
