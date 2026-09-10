"""Tests for control flow optimizations: branch pruning, invariant hoisting, loop unrolling, and parallel scan."""

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.ops.control_flow.cond import cond
from ml_switcheroo_compiler.ops.control_flow.scan import scan
from ml_switcheroo_compiler.ops.control_flow.while_loop import while_loop
from ml_switcheroo_compiler.transforms.passes.control_flow_opt import (
    control_flow_optimization_pass,
    evaluate_static_predicate,
)
from ml_switcheroo_compiler.transforms.passes.loop_unrolling import (
    loop_unrolling_pass,
)
from ml_switcheroo_compiler.transforms.passes.parallel_scan import (
    _load_scan_rules,
    detect_associative_reduction,
    parallel_scan_pass,
)


def test_evaluate_static_predicate():
    """Test static predicate evaluation."""
    g = IRGraph()
    # Missing node
    assert evaluate_static_predicate(g, "missing") is None

    # Constant boolean
    g.nodes["c_true"] = IRNode(id="c_true", op_type="Constant", attributes={"value": True})
    assert evaluate_static_predicate(g, "c_true") is True

    g.nodes["c_false"] = IRNode(id="c_false", op_type="Constant", attributes={"value": False})
    assert evaluate_static_predicate(g, "c_false") is False

    # Comparison constants
    g.nodes["c_1"] = IRNode(id="c_1", op_type="Constant", attributes={"value": 1})
    g.nodes["c_2"] = IRNode(id="c_2", op_type="Constant", attributes={"value": 2})

    g.nodes["eq"] = IRNode(id="eq", op_type="Equal", inputs=["c_1", "c_1"])
    assert evaluate_static_predicate(g, "eq") is True

    g.nodes["lt"] = IRNode(id="lt", op_type="Less", inputs=["c_1", "c_2"])
    assert evaluate_static_predicate(g, "lt") is True

    g.nodes["gt"] = IRNode(id="gt", op_type="Greater", inputs=["c_1", "c_2"])
    assert evaluate_static_predicate(g, "gt") is False

    g.nodes["le"] = IRNode(id="le", op_type="LessEqual", inputs=["c_1", "c_1"])
    assert evaluate_static_predicate(g, "le") is True

    g.nodes["ge"] = IRNode(id="ge", op_type="GreaterEqual", inputs=["c_1", "c_2"])
    assert evaluate_static_predicate(g, "ge") is False


def test_branch_pruning_true():
    """Test pruning false branch when predicate is statically True."""
    g = IRGraph(name="test_cond_prune")
    g.nodes["pred"] = IRNode(id="pred", op_type="Constant", attributes={"value": True})

    true_b = IRGraph(name="tb")
    true_b.inputs = ["t_in"]
    true_b.nodes["t1"] = IRNode(id="t1", op_type="Add", inputs=["t_in", "t_in"])
    true_b.outputs = ["t1"]

    false_b = IRGraph(name="fb")
    false_b.inputs = ["f_in"]
    false_b.nodes["f1"] = IRNode(id="f1", op_type="Subtract", inputs=["f_in", "f_in"])
    false_b.outputs = ["f1"]

    g.nodes["cond_node"] = IRNode(
        id="cond_node",
        op_type="Cond",
        inputs=["pred"],
        attributes={"then_branch": true_b, "else_branch": false_b},
    )

    opt = control_flow_optimization_pass(g)
    assert opt.nodes["cond_node"].op_type == "Identity"
    assert any("t1" in nid for nid in opt.nodes)
    assert not any("f1" in nid for nid in opt.nodes)


def test_branch_pruning_false():
    """Test pruning true branch when predicate is statically False."""
    g = IRGraph(name="test_cond_prune_false")
    g.nodes["pred"] = IRNode(id="pred", op_type="Constant", attributes={"value": False})

    true_b = IRGraph(name="tb")
    true_b.nodes["t1"] = IRNode(id="t1", op_type="Add", inputs=["x", "x"])
    true_b.outputs = ["t1"]

    false_b = IRGraph(name="fb")
    false_b.nodes["f1"] = IRNode(id="f1", op_type="Multiply", inputs=["x", "x"])
    false_b.outputs = ["f1"]

    g.nodes["cond_node"] = IRNode(
        id="cond_node",
        op_type="If",
        inputs=["pred"],
        attributes={"then_branch": true_b, "else_branch": false_b},
    )

    opt = control_flow_optimization_pass(g)
    assert opt.nodes["cond_node"].op_type == "Identity"
    assert any("f1" in nid for nid in opt.nodes)
    assert not any("t1" in nid for nid in opt.nodes)


def test_invariant_hoisting():
    """Test hoisting common invariant computation from branches."""
    g = IRGraph(name="outer")
    g.nodes["ext_a"] = IRNode(id="ext_a", op_type="Input", inputs=[])
    g.nodes["ext_b"] = IRNode(id="ext_b", op_type="Input", inputs=[])
    g.nodes["dyn_pred"] = IRNode(id="dyn_pred", op_type="Input", inputs=[])

    true_b = IRGraph(name="tb")
    true_b.nodes["t_common"] = IRNode(id="t_common", op_type="Add", inputs=["ext_a", "ext_b"])
    true_b.nodes["t_unique"] = IRNode(id="t_unique", op_type="Exp", inputs=["t_common"])
    true_b.outputs = ["t_unique"]

    false_b = IRGraph(name="fb")
    false_b.nodes["f_common"] = IRNode(id="f_common", op_type="Add", inputs=["ext_a", "ext_b"])
    false_b.nodes["f_unique"] = IRNode(id="f_unique", op_type="Log", inputs=["f_common"])
    false_b.outputs = ["f_unique"]

    g.nodes["cond1"] = IRNode(
        id="cond1",
        op_type="Cond",
        inputs=["dyn_pred"],
        attributes={"then_branch": true_b, "else_branch": false_b},
    )

    opt = control_flow_optimization_pass(g)
    assert any("hoisted_add" in nid for nid in opt.nodes)


def test_fori_loop_unrolling():
    """Test unrolling ForiLoop when upper and lower bounds are statically known."""
    body = IRGraph(name="body")
    body.inputs = ["iter_i", "acc"]
    body.nodes["iter_i"] = IRNode(id="iter_i", op_type="Input", inputs=[])
    body.nodes["acc"] = IRNode(id="acc", op_type="Input", inputs=[])
    body.nodes["add_acc"] = IRNode(id="add_acc", op_type="Add", inputs=["acc", "iter_i"])
    body.outputs = ["add_acc"]

    g = IRGraph(name="fori_test")
    g.inputs = ["init_val"]
    g.nodes["init_val"] = IRNode(id="init_val", op_type="Input", inputs=[])
    g.nodes["fori"] = IRNode(
        id="fori",
        op_type="ForiLoop",
        inputs=["init_val"],
        attributes={"lower": 0, "upper": 3, "body": body},
    )
    g.outputs = ["fori"]

    opt = loop_unrolling_pass(g)
    assert opt.nodes["fori"].op_type == "Identity"
    # Should have 3 unrolled iterations (iter0, iter1, iter2) with injected constant counter
    assert any("const_idx" in nid for nid in opt.nodes)
    assert any("fori_iter0" in nid for nid in opt.nodes)
    assert any("fori_iter1" in nid for nid in opt.nodes)
    assert any("fori_iter2" in nid for nid in opt.nodes)


def test_parallel_scan_pass():
    """Test parallel scan optimization converting sequential scan into CumSum."""
    body = IRGraph(name="scan_body")
    body.inputs = ["carry", "x"]
    body.nodes["carry"] = IRNode(id="carry", op_type="Input", inputs=[])
    body.nodes["x"] = IRNode(id="x", op_type="Input", inputs=[])
    body.nodes["acc"] = IRNode(id="acc", op_type="Add", inputs=["carry", "x"])
    body.outputs = ["acc"]

    g = IRGraph(name="scan_graph")
    g.inputs = ["init_c", "xs"]
    g.nodes["init_c"] = IRNode(id="init_c", op_type="Input", inputs=[])
    g.nodes["xs"] = IRNode(id="xs", op_type="Input", inputs=[], shape_metadata=(10, 4))
    g.nodes["scan1"] = IRNode(
        id="scan1",
        op_type="Scan",
        inputs=["init_c", "xs"],
        attributes={"body": body},
        shape_metadata=(10, 4),
    )
    g.outputs = ["scan1"]

    opt = parallel_scan_pass(g)
    assert opt.nodes["scan1"].op_type == "Identity"
    assert any("parallel_cumsum" in nid for nid in opt.nodes)
    assert any(n.op_type == "CumSum" for n in opt.nodes.values())


def test_control_flow_ops_exports():
    """Test dedicated module imports for control flow operators."""
    assert callable(cond)
    assert callable(while_loop)
    assert callable(scan)
    assert isinstance(_load_scan_rules(), dict)
    assert detect_associative_reduction(None, {}) is None


def test_control_flow_opt_exhaustive_branches():
    """Verify remaining branch edge cases in control_flow_opt."""
    g = IRGraph()
    # 1. Constant with string value (hasattr(val, 'item') is False)
    g.nodes["c_str"] = IRNode(id="c_str", op_type="Constant", attributes={"value": "hello"})
    assert evaluate_static_predicate(g, "c_str") is None

    # 2. Comparison op with fewer than 2 inputs
    g.nodes["eq_single"] = IRNode(id="eq_single", op_type="Equal", inputs=["c_str"])
    assert evaluate_static_predicate(g, "eq_single") is None

    # 3. Comparison op with non-number constant inputs
    g.nodes["c_str2"] = IRNode(id="c_str2", op_type="Constant", attributes={"value": "world"})
    g.nodes["eq_str"] = IRNode(id="eq_str", op_type="Equal", inputs=["c_str", "c_str2"])
    assert evaluate_static_predicate(g, "eq_str") is None

    # 4. static_pred is False but false_branch is None (active_branch is None)
    g_false_none = IRGraph(name="false_none")
    g_false_none.nodes["pred_f"] = IRNode(id="pred_f", op_type="Constant", attributes={"value": False})
    true_b = IRGraph(name="tb")
    true_b.nodes["t1"] = IRNode(id="t1", op_type="Add", inputs=["x", "x"])
    g_false_none.nodes["cond_no_f"] = IRNode(
        id="cond_no_f",
        op_type="Cond",
        inputs=["pred_f"],
        attributes={"then_branch": true_b, "else_branch": None},
    )
    opt_f = control_flow_optimization_pass(g_false_none)
    assert "cond_no_f" in opt_f.nodes

    # 5. Dynamic pred with true and false branches having nothing to hoist
    g_no_hoist = IRGraph(name="no_hoist")
    g_no_hoist.nodes["dyn_p"] = IRNode(id="dyn_p", op_type="Input", inputs=[])
    fb_diff = IRGraph(name="fb")
    fb_diff.nodes["f_diff"] = IRNode(id="f_diff", op_type="Multiply", inputs=["a", "b"])
    g_no_hoist.nodes["cond_diff"] = IRNode(
        id="cond_diff",
        op_type="Cond",
        inputs=["dyn_p"],
        attributes={"then_branch": true_b, "else_branch": fb_diff},
    )
    opt_no_hoist = control_flow_optimization_pass(g_no_hoist)
    assert "cond_diff" in opt_no_hoist.nodes


def test_parallel_scan_exhaustive_branches():
    """Verify remaining branch edge cases in parallel_scan."""
    from unittest.mock import patch

    # 1. _load_scan_rules when safe_load returns non-dict
    with patch("yaml.safe_load", return_value=["not", "a", "dict"]):
        assert _load_scan_rules() == {}

    # 2. detect_associative_reduction with non-associative op
    body_non_assoc = IRGraph(name="body_non_assoc")
    body_non_assoc.nodes["in0"] = IRNode(id="in0", op_type="Input", inputs=[])
    body_non_assoc.nodes["sin0"] = IRNode(id="sin0", op_type="Sin", inputs=["in0"])
    body_non_assoc.outputs = ["sin0"]
    assert detect_associative_reduction(body_non_assoc, {"associative_ops": {"Add": {}}}) is None

    # 3. parallel_scan_pass where reduction_info is None
    g_scan_non_assoc = IRGraph(name="scan_non_assoc")
    g_scan_non_assoc.nodes["in0"] = IRNode(id="in0", op_type="Input", inputs=[])
    g_scan_non_assoc.nodes["scan0"] = IRNode(
        id="scan0",
        op_type="Scan",
        inputs=["in0"],
        attributes={"body": body_non_assoc},
    )
    opt_non_assoc = parallel_scan_pass(g_scan_non_assoc)
    assert opt_non_assoc.nodes["scan0"].op_type == "Scan"

    # 4. parallel_scan_pass with no Scan nodes (transformed is False)
    g_no_scan = IRGraph(name="no_scan")
    g_no_scan.nodes["add0"] = IRNode(id="add0", op_type="Add", inputs=["a", "b"])
    opt_no_scan = parallel_scan_pass(g_no_scan)
    assert "add0" in opt_no_scan.nodes
