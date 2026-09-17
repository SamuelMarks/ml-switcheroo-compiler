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
    """Test _inline_subgraph helper."""
    g = LogicalGraph()
    sg = LogicalGraph()
    n = LogicalNode(id="n", op_type="Custom")
    sg.nodes["i"] = LogicalNode(id="i", op_type="Input")
    sg.nodes["a"] = LogicalNode(id="a", op_type="Add", inputs=["i"])
    _inline_subgraph(g, sg, n, {"i": "x", "a": "new_a"})
    assert "new_a" in g.nodes


def test_inline_grad_subgraph():
    """Test _inline_grad_subgraph helper."""
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
    """Test _if_vjp."""
    g = LogicalGraph()
    n = LogicalNode(id="n", op_type="If", inputs=["cond", "a", "b"], attributes={"true_branch": LogicalGraph(), "false_branch": LogicalGraph()})
    res = _if_vjp(g, n, "cot")
    assert len(res) >= 1


def test_loop_vjp():
    """Test _loop_vjp."""
    g = LogicalGraph()
    n = LogicalNode(id="n", op_type="Loop", inputs=["init", "n_iter"], attributes={"body": LogicalGraph()})
    res = _loop_vjp(g, n, "cot")
    assert len(res) >= 1


def test_scan_vjp():
    """Test _scan_vjp."""
    g = LogicalGraph()
    n = LogicalNode(id="n", op_type="Scan", inputs=["a", "b"], attributes={"body": LogicalGraph()})
    res = _scan_vjp(g, n, "cot")
    assert len(res) >= 1


def test_assoc_scan_vjp():
    """Test _assoc_scan_vjp."""
    g = LogicalGraph()
    n = LogicalNode(id="n", op_type="AssociativeScan", inputs=["a", "b", "c"], attributes={"combine_fn": LogicalGraph()})
    res = _assoc_scan_vjp(g, n, "cot")
    assert len(res) >= 1


def test_jvps():
    """Test control flow JVPs."""
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
    """Placeholder for null tests."""
    pass


def test_checkpoint_vjp():
    """Test checkpoint_vjp with subgraph."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import checkpoint_vjp

    sg = IRGraph()
    sg.nodes["inp"] = IRNode(id="inp", op_type="Input")
    sg.nodes["out"] = IRNode(id="out", op_type="Exp", inputs=["inp"])
    sg.inputs = ["inp"]
    sg.outputs = ["out"]

    node = IRNode(id="cp", op_type="Checkpoint", inputs=["in_main"], attributes={"subgraph": sg})

    main_graph = IRGraph()
    main_graph.nodes["in_main"] = IRNode(id="in_main", op_type="Input")

    with patch("ml_switcheroo_compiler.transforms.autodiff.grad") as mock_grad:
        sg_grad = IRGraph()
        sg_grad.nodes["out"] = IRNode(id="out", op_type="Input")
        sg_grad.nodes["inp"] = IRNode(id="inp", op_type="Output", inputs=["out"])
        sg_grad.nodes["mid"] = IRNode(id="mid", op_type="Exp", inputs=["out"])
        sg_grad.nodes["cotangent_id"] = IRNode(id="cotangent_id", op_type="Exp")
        sg_grad.inputs = ["out"]
        sg_grad.outputs = ["inp"]
        mock_grad.return_value = sg_grad

        res = checkpoint_vjp(main_graph, node, "cotangent_id")
        assert res is not None


def test_if_jvp():
    """Test if_jvp and recompute_vjp."""
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
        mock_jvp.return_value = IRGraph()

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
    """Test custom_vjp_vjp with bwd_fn."""
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import custom_vjp_vjp

    graph = IRGraph()
    node = IRNode(id="test_node", op_type="CustomVJP", inputs=["a", "b"], attributes={"bwd_fn": "fake_bwd_fn"})
    node.shape_metadata = ()

    res = custom_vjp_vjp(graph, node, "cot")
    assert len(res) == 2
    assert "ProcessCustomVJPCall" in [n.op_type for n in graph.nodes.values()]
    assert "TupleGetItem" in [n.op_type for n in graph.nodes.values()]


def test_custom_vjp_rules_empty_attributes() -> None:
    """Test _cond_vjp, _scan_vjp, _while_loop_vjp, _cond_jvp, and _while_loop_jvp with empty attributes."""
    from ml_switcheroo_compiler.ir.core import IRGraph
    from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import (
        _cond_jvp,
        _cond_vjp,
        _scan_vjp,
        _while_loop_jvp,
        _while_loop_vjp,
    )

    graph = IRGraph(name="vjp_test")
    node = LogicalNode(id="flow_op", op_type="Cond", inputs=["in_cond", "in_data"], attributes={})
    cond_res = _cond_vjp(graph, node, "ct")
    assert len(cond_res) == 2
    scan_res = _scan_vjp(graph, node, "ct")
    assert len(scan_res) == 2
    while_vjp_res = _while_loop_vjp(graph, node, "ct")
    assert len(while_vjp_res) == 2
    while_jvp_res = _while_loop_jvp(graph, node, "t")
    assert while_jvp_res is not None
    cond_jvp_res = _cond_jvp(graph, node, ["t1", "t2"])
    assert cond_jvp_res == "mock_tangent"


def test_checkpoint_vjp_with_fun():
    """Test checkpoint_vjp when fun is provided in node.attributes."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import checkpoint_vjp

    def sample_fn(x):
        """Identity sample function for tracing."""
        return x

    main_graph = IRGraph()
    main_graph.nodes["x"] = IRNode(id="x", op_type="Input", shape_metadata=(2, 2))
    node = IRNode(id="cp_fun", op_type="Checkpoint", inputs=["x"], attributes={"fun": sample_fn})

    with patch("ml_switcheroo_compiler.transforms.autodiff.grad") as mock_grad:
        sg_grad = IRGraph()
        sg_grad.nodes["out"] = IRNode(id="out", op_type="Input")
        sg_grad.nodes["in_node"] = IRNode(id="in_node", op_type="Output", inputs=["out"])
        sg_grad.inputs = ["out"]
        sg_grad.outputs = ["in_node"]
        mock_grad.return_value = sg_grad

        res = checkpoint_vjp(main_graph, node, "cot")
        assert res is not None
        assert len(res) == 1


def test_cond_vjp_with_branches():
    """Test _cond_vjp when both branches are provided."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import _cond_vjp

    tb = IRGraph()
    tb.inputs = ["t_in"]
    tb.outputs = ["t_out"]
    eb = IRGraph()
    eb.inputs = ["e_in"]
    eb.outputs = ["e_out"]

    graph = IRGraph()
    node = IRNode(id="cond_node", op_type="Cond", inputs=["pred", "x"], attributes={"then_branch": tb, "else_branch": eb})

    with patch("ml_switcheroo_compiler.transforms.autodiff.grad") as mock_grad:
        mock_grad.return_value = IRGraph()
        res = _cond_vjp(graph, node, "cot")
        assert len(res) == 2
        assert f"{node.id}_adj_cond" in graph.nodes


def test_scan_vjp_with_body():
    """Test _scan_vjp when body subgraph is provided."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import _scan_vjp

    body = IRGraph()
    body.inputs = ["b_in"]
    body.outputs = ["b_out"]

    graph = IRGraph()
    node = IRNode(id="scan_node", op_type="Scan", inputs=["init", "xs"], attributes={"body": body, "reverse": False})

    with patch("ml_switcheroo_compiler.transforms.autodiff.grad") as mock_grad:
        mock_grad.return_value = IRGraph()
        res = _scan_vjp(graph, node, "cot")
        assert len(res) == 2
        assert f"{node.id}_adj_scan" in graph.nodes


def test_while_loop_vjp_with_body():
    """Test _while_loop_vjp when body subgraph is provided."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import _while_loop_vjp

    body = IRGraph()
    body.inputs = ["b_in"]
    body.outputs = ["b_out"]

    graph = IRGraph()
    node = IRNode(id="while_node", op_type="WhileLoop", inputs=["init"], attributes={"body": body, "cond": IRGraph()})

    with patch("ml_switcheroo_compiler.transforms.autodiff.grad") as mock_grad:
        mock_grad.return_value = IRGraph()
        res = _while_loop_vjp(graph, node, "cot")
        assert len(res) == 1
        assert f"{node.id}_adj_while" in graph.nodes


def test_cond_jvp_with_branches():
    """Test _cond_jvp when both branches are provided."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import _cond_jvp

    tb = IRGraph()
    tb.inputs = ["t_in"]
    tb.outputs = ["t_out"]
    eb = IRGraph()
    eb.inputs = ["e_in"]
    eb.outputs = ["e_out"]

    graph = IRGraph()
    node = IRNode(id="cond_node", op_type="Cond", inputs=["pred", "x"], attributes={"then_branch": tb, "else_branch": eb})

    with patch("ml_switcheroo_compiler.transforms.autodiff.jvp") as mock_jvp:
        mock_jvp.return_value = IRGraph()
        res = _cond_jvp(graph, node, ["t_pred", "t_x"])
        assert res == f"{node.id}_jvp"
        assert res in graph.nodes


def test_while_loop_jvp_with_body():
    """Test _while_loop_jvp when body is provided."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import _while_loop_jvp

    body = IRGraph()
    body.inputs = ["b_in"]
    body.outputs = ["b_out"]

    graph = IRGraph()
    node = IRNode(id="while_node", op_type="WhileLoop", inputs=["x"], attributes={"body": body, "cond": "mock_cond"})

    with patch("ml_switcheroo_compiler.transforms.autodiff.jvp") as mock_jvp:
        mock_jvp.return_value = IRGraph()
        res = _while_loop_jvp(graph, node, ["t_x"])
        assert res == f"{node.id}_jvp"
        assert res in graph.nodes


def test_scan_jvp_with_body():
    """Test _scan_jvp when body is provided."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import _scan_jvp

    body = IRGraph()
    body.inputs = ["b_in"]
    body.outputs = ["b_out"]

    graph = IRGraph()
    node = IRNode(id="scan_node", op_type="Scan", inputs=["x"], attributes={"body": body, "reverse": True})

    with patch("ml_switcheroo_compiler.transforms.autodiff.jvp") as mock_jvp:
        mock_jvp.return_value = IRGraph()
        res = _scan_jvp(graph, node, ["t_x"])
        assert res == f"{node.id}_jvp"
        assert res in graph.nodes


def test_custom_vjp_vjp_with_bwd_subgraph():
    """Test custom_vjp_vjp when bwd_subgraph is provided."""
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import custom_vjp_vjp

    # Test case 1: bwd_subgraph has inputs and nodes dict, plus extra inputs to cover i >= len(node.inputs)
    bwd_sg = IRGraph()
    bwd_sg.inputs = ["cot_in", "x_in", "extra_in"]
    bwd_sg.nodes["cot_in"] = IRNode(id="cot_in", op_type="Input")
    bwd_sg.nodes["x_in"] = IRNode(id="x_in", op_type="Input")
    bwd_sg.nodes["extra_in"] = IRNode(id="extra_in", op_type="Input")
    bwd_sg.nodes["grad_x"] = IRNode(id="grad_x", op_type="Mul", inputs=["cot_in", "x_in"])
    bwd_sg.outputs = ["grad_x"]

    main_graph = IRGraph()
    node = IRNode(id="cvjp_node", op_type="CustomVJP", inputs=["x_main"], attributes={"bwd_graph": bwd_sg})

    res = custom_vjp_vjp(main_graph, node, "cot_main")
    assert len(res) == 1

    # Test case 2: bwd_subgraph has empty inputs and list of nodes
    class DummyGraphWithList:
        """Dummy graph container with a list of nodes."""

        def __init__(self):
            """Initialize dummy graph."""
            self.inputs = []
            self.outputs = []
            self.nodes = [IRNode(id="n1", op_type="Exp", inputs=[])]

    dummy_sg = DummyGraphWithList()
    node2 = IRNode(id="cvjp_node2", op_type="CustomVJP", inputs=[], attributes={"custom_backward_subgraph": dummy_sg})
    res2 = custom_vjp_vjp(main_graph, node2, "cot2")
    assert len(res2) == 0


def test_custom_jvp_jvp():
    """Test custom_jvp_jvp with subgraph and with fallback rule."""
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import custom_jvp_jvp

    # Subgraph path with outputs
    jvp_sg = IRGraph()
    jvp_sg.inputs = ["x_in", "tx_in"]
    jvp_sg.nodes["x_in"] = IRNode(id="x_in", op_type="Input")
    jvp_sg.nodes["tx_in"] = IRNode(id="tx_in", op_type="Input")
    jvp_sg.nodes["out_tan"] = IRNode(id="out_tan", op_type="Identity", inputs=["tx_in"])
    jvp_sg.outputs = ["out_tan"]

    main_g = IRGraph()
    node = IRNode(id="cjvp_node", op_type="CustomJVP", inputs=["x"], attributes={"jvp_graph": jvp_sg})
    res = custom_jvp_jvp(main_g, node, ["t_x"])
    assert res is not None

    # Subgraph path with empty outputs and list of nodes, single tangent
    class DummyJVPGraphList:
        """Dummy JVP graph container with a list of nodes."""

        def __init__(self):
            """Initialize dummy JVP graph."""
            self.inputs = ["x1", "x2", "extra"]
            self.outputs = []
            self.nodes = [IRNode(id="k1", op_type="Identity", inputs=[])]

    node2 = IRNode(id="cjvp_node2", op_type="CustomJVP", inputs=["x1"], attributes={"custom_forward_subgraph": DummyJVPGraphList()})
    res2 = custom_jvp_jvp(main_g, node2, "t_single")
    assert res2 == "t_single"

    # Fallback path (no subgraph)
    node_fb = IRNode(id="cjvp_fb", op_type="CustomJVP", inputs=["x"], attributes={"jvp_rule": "mock_rule", "fun": "mock_fun"})
    res_fb = custom_jvp_jvp(main_g, node_fb, ["t_x"])
    assert "cjvp_tan_" in res_fb
    assert res_fb in main_g.nodes


def test_custom_jvp_vjp():
    """Test custom_jvp_vjp with fun and fallback without fun."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import custom_jvp_vjp

    def sample_fn(x):
        """Identity function for custom JVP VJP test."""
        return x

    main_g = IRGraph()
    main_g.nodes["x"] = IRNode(id="x", op_type="Input", shape_metadata=(1,))
    node = IRNode(id="cjvp_node", op_type="CustomJVP", inputs=["x"], attributes={"fun": sample_fn})

    with patch("ml_switcheroo_compiler.transforms.autodiff.grad") as mock_grad:
        sg_grad = IRGraph()
        sg_grad.nodes["out"] = IRNode(id="out", op_type="Input")
        sg_grad.nodes["in_node"] = IRNode(id="in_node", op_type="Output", inputs=["out"])
        sg_grad.inputs = ["out"]
        sg_grad.outputs = ["in_node"]
        mock_grad.return_value = sg_grad

        res = custom_jvp_vjp(main_g, node, "cot")
        assert len(res) == 1

    # Fallback without fun
    node_nofun = IRNode(id="cjvp_nofun", op_type="CustomJVP", inputs=["x", "y"], attributes={})
    res_nofun = custom_jvp_vjp(main_g, node_nofun, "cot")
    assert res_nofun == ("cot", "cot")


def test_indexing_and_zeros_autodiff_rules():
    """Test getitem_vjp, index_put_jvp, and zeros_like_jvp."""
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import getitem_vjp, index_put_jvp, zeros_like_jvp

    g = IRGraph()
    g.nodes["arr"] = IRNode(id="arr", op_type="Input", shape_metadata=(4, 4))
    node_get = IRNode(id="get_node", op_type="GetItem", inputs=["arr"], attributes={"key": 2})
    res_get = getitem_vjp(g, node_get, "cot")
    assert len(res_get) == 1
    assert res_get[0] in g.nodes

    # getitem with no inputs
    node_get_no_in = IRNode(id="get_no_in", op_type="GetItem", inputs=[])
    res_get2 = getitem_vjp(g, node_get_no_in, "cot")
    assert len(res_get2) == 1

    # index_put_jvp with varying tangent lengths
    node_put = IRNode(id="put_node", op_type="IndexPut", inputs=["arr", "val"], attributes={"key": 1})
    res_put2 = index_put_jvp(g, node_put, ["t_arr", "t_val"])
    assert res_put2 in g.nodes
    res_put1 = index_put_jvp(g, node_put, ["t_arr"])
    assert res_put1 in g.nodes
    res_put0 = index_put_jvp(g, node_put, [])
    assert res_put0 in g.nodes
    res_put_single = index_put_jvp(g, node_put, "t_single")
    assert res_put_single in g.nodes

    # zeros_like_jvp
    node_zeros = IRNode(id="z_node", op_type="ZerosLike", inputs=["arr"])
    res_z = zeros_like_jvp(g, node_zeros, ["t_arr"])
    assert res_z in g.nodes
