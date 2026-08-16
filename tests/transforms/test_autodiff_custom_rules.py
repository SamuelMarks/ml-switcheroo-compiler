"""Coverage tests for custom rules autodiff."""

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients
from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import (
    _assoc_scan_vjp,
    _if_vjp,
    _loop_vjp,
    _scan_vjp,
)


def test_if_vjp():
    assert _if_vjp(None, None, None) == (UnconnectedGradients.ZERO,)


def test_make_zero():
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import make_zero_jvp, make_zero_vjp

    node = LogicalNode(id="n1", op_type="Loop", inputs=["a", "b"])
    vjp = make_zero_vjp("Test")
    assert vjp(None, node, None) == (UnconnectedGradients.ZERO, UnconnectedGradients.ZERO)

    jvp = make_zero_jvp("Test")
    assert jvp(None, node, None) == ""


def test_loop_vjp():
    node = LogicalNode(id="n1", op_type="Loop", inputs=["a", "b"])
    assert _loop_vjp(None, node, None) == (UnconnectedGradients.ZERO, UnconnectedGradients.ZERO)


def test_scan_vjp():
    node = LogicalNode(id="n1", op_type="Scan", inputs=["a"])
    assert _scan_vjp(None, node, None) == (UnconnectedGradients.ZERO,)


def test_assoc_scan_vjp():
    node = LogicalNode(id="n1", op_type="AssociativeScan", inputs=["a", "b", "c"])
    assert _assoc_scan_vjp(None, node, None) == (UnconnectedGradients.ZERO, UnconnectedGradients.ZERO, UnconnectedGradients.ZERO)


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
