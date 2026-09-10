from unittest import mock

import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.errors import MissingJVPRuleError
from ml_switcheroo_compiler.transforms.autodiff import (
    _accumulate_gradients,
    _add_nodes,
    _backward_pass,
    _compile_jvp_expr,
    _copy_graph,
    _extract_gradients,
    _forward_pass_jvp,
    _get_input_tangents,
    _get_reachable_from_output,
    _invoke_jvp_rule,
    _invoke_style2_jvp_rule,
    _process_jvp_node,
    _recompute_subgraph,
    grad,
    hvp,
    jvp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients


def test_add_nodes():
    graph = LogicalGraph()
    graph.nodes["n1"] = LogicalNode("n1", "Input", shape_metadata=())
    graph.nodes["n2"] = LogicalNode("n2", "Input", shape_metadata=())
    out_id = _add_nodes(graph, "n1", "n2")
    assert out_id in graph.nodes
    assert graph.nodes[out_id].op_type == "Add"


def test_copy_graph():
    graph = LogicalGraph("test")
    graph.nodes["n1"] = LogicalNode("n1", "Input")
    graph2 = _copy_graph(graph)
    assert graph2.name == "test_grad"
    assert "n1" in graph2.nodes


def test_get_reachable_from_output():
    n1 = LogicalNode("n1", "Input")
    n2 = LogicalNode("n2", "Add", inputs=["n1"])
    reachable = _get_reachable_from_output([n1, n2], "n2")
    assert "n1" in reachable
    assert "n2" in reachable


def test_recompute_subgraph():
    graph = LogicalGraph()
    n1 = LogicalNode("n1", "Input", attributes={"rematerialize": True})
    n2 = LogicalNode("n2", "Add", inputs=["n1"])
    graph.nodes["n1"] = n1
    graph.nodes["n2"] = n2

    n2_re = _recompute_subgraph(graph, n2)
    assert n2_re.id != "n2"
    assert len(n2_re.inputs) == 1
    assert n2_re.inputs[0] != "n1"

    n3 = LogicalNode("n3", "Add", inputs=["missing"])
    n3_re = _recompute_subgraph(graph, n3)
    assert n3_re.inputs[0] == "missing"

    n4 = LogicalNode("n4", "Add", inputs=["n2"])  # no remat
    n4_re = _recompute_subgraph(graph, n4)
    assert n4_re.inputs[0] == "n2"


def test_accumulate_gradients():
    graph = LogicalGraph()
    n1 = LogicalNode("n1", "Add", inputs=["i1", "i2"])
    n1.attributes["rematerialize"] = True
    graph.nodes["n1"] = n1
    n1_no_remat = LogicalNode("n1_no_remat", "Add", inputs=["i1", "i2"])
    graph.nodes["n1_no_remat"] = n1_no_remat
    graph.nodes["adj_i1"] = LogicalNode("adj_i1", "Constant")
    graph.nodes["adj_i2"] = LogicalNode("adj_i2", "Constant")

    with mock.patch("ml_switcheroo_compiler.transforms.autodiff.get_vjp") as mock_get_vjp:
        mock_get_vjp.return_value = lambda g, n, adj: ["adj_i1", "adj_i2"]
        adjoints = {}
        _accumulate_gradients(graph, n1, "adj_n1", adjoints)
        assert adjoints["i1"] == "adj_i1"

        _accumulate_gradients(graph, n1, "adj_n1", adjoints)
        assert adjoints["i1"].startswith("adj_i1_add")

        # Test Unconnected
        mock_get_vjp.return_value = lambda g, n, adj: [UnconnectedGradients.NONE, UnconnectedGradients.ZERO]
        adjoints2 = {}
        _accumulate_gradients(graph, n1, "adj_n1", adjoints2)
        _accumulate_gradients(graph, n1_no_remat, "adj_n1", adjoints2)
        assert "i1" not in adjoints2

        # Test Missing VJP
        mock_get_vjp.side_effect = ValueError("missing")
        with pytest.raises(MissingJVPRuleError):
            _accumulate_gradients(graph, n1, "adj_n1", adjoints2)

        # Test length mismatch
        mock_get_vjp.side_effect = None
        mock_get_vjp.return_value = lambda g, n, adj: ["adj_i1"]
        with pytest.raises(ValueError):
            _accumulate_gradients(graph, n1, "adj_n1", adjoints2)


def test_backward_pass():
    graph = LogicalGraph()
    n1 = LogicalNode("n1", "Output", inputs=["n2"])
    n2 = LogicalNode("n2", "Add", inputs=["n3"])
    n3 = LogicalNode("n3", "Input", inputs=[])
    n4 = LogicalNode("n4", "Output", inputs=["n2"])
    n5 = LogicalNode("n5", "Unknown", inputs=[])
    graph.nodes = {"n1": n1, "n2": n2, "n3": n3, "n4": n4}

    adjoints = {"n1": "adj_n1", "n3": "adj_n3"}
    with mock.patch("ml_switcheroo_compiler.transforms.autodiff._accumulate_gradients") as mock_acc:
        _backward_pass(graph, [n5, n4, n3, n2, n1], {"n1", "n2", "n3", "n4"}, adjoints)
        assert adjoints["n2"] == "adj_n1"  # output propagates directly
        mock_acc.assert_called_once()  # Called for n2, n3 is Input (skipped), n1 is Output, n4 is unreachable


def test_extract_gradients():
    graph = LogicalGraph()
    graph.nodes["n1"] = LogicalNode("n1", "Input", shape_metadata=())
    adjoints = {"n1": "adj_n1", "n3": "adj_n3"}
    grads = _extract_gradients(graph, ["n1"], adjoints)
    assert grads == ["adj_n1"]

    # Not in adjoints -> adds a zero node
    graph.nodes["n2"] = LogicalNode("n2", "Input", shape_metadata=())
    grads2 = _extract_gradients(graph, ["n2"], adjoints)
    assert grads2[0].startswith("grad_zeros")

    with pytest.raises(ValueError):
        _extract_gradients(graph, ["n3"], adjoints)


def test_grad_func():
    graph = LogicalGraph()
    graph.nodes["n1"] = LogicalNode("n1", "Input", shape_metadata=())
    graph.nodes["n2"] = LogicalNode("n2", "Add", inputs=["n1"], shape_metadata=())

    with pytest.raises(ValueError):
        grad(graph, ["n1"], "missing")

    with mock.patch("ml_switcheroo_compiler.transforms.autodiff._backward_pass"):
        g_out = grad(graph, ["n1"], "n2")
        assert len(g_out.outputs) == 1
        assert "grad_ones" in g_out.outputs[0] or "grad_ones" in str(g_out.nodes.keys())

        # With dict cotangent
        g_out2 = grad(graph, ["n1"], "n2", cotangent_id={"n2": "my_cot"})
        g_out3 = grad(graph, ["n1"], "n2", cotangent_id="my_cot")
        assert g_out2 is not None


def test_get_input_tangents():
    graph = LogicalGraph()
    n1 = LogicalNode("n1", "Add", inputs=["i1", "i2"])
    n1.attributes["rematerialize"] = True
    graph.nodes["i1"] = LogicalNode("i1", "Input", shape_metadata=())
    graph.nodes["i2"] = LogicalNode("i2", "Input", shape_metadata=())

    tangents = {"i1": "t1"}
    t_in = _get_input_tangents(graph, n1, tangents)
    assert t_in[0] == "t1"
    assert t_in[1].startswith("jvp_zeros")


def test_compile_jvp_expr():
    graph = LogicalGraph()
    inverse_map = {"a": "orig_a", "b": "orig_b"}

    # Test Name
    assert _compile_jvp_expr("a", graph, (), inverse_map) == "orig_a"

    # Test Unary USub
    # emit_ir_node doesn't exist, we need to mock it
    with mock.patch("ml_switcheroo_compiler.ops.base.emit_ir_node", return_value="out_neg") as mock_emit:
        assert _compile_jvp_expr("-a", graph, (), inverse_map) == "out_neg"
        mock_emit.assert_called_with(graph, "Negative", ["orig_a"], ())

    # Test Binary
    with mock.patch("ml_switcheroo_compiler.ops.base.emit_ir_node", return_value="out_bin") as mock_emit:
        assert _compile_jvp_expr("a + b", graph, (), inverse_map) == "out_bin"
        mock_emit.assert_called_with(graph, "Add", ["orig_a", "orig_b"], ())

    # Test Const
    const_id = _compile_jvp_expr("5", graph, (), inverse_map)
    assert const_id.startswith("jvp_const")

    # Errors
    with pytest.raises(ValueError):
        _compile_jvp_expr("a ** b", graph, (), inverse_map)  # Pow not in op_map

    assert _compile_jvp_expr("~a", graph, (), inverse_map) == "orig_a"
    assert _compile_jvp_expr("+a", graph, (), inverse_map) == "orig_a"

    with pytest.raises(ValueError):
        _compile_jvp_expr("a.b", graph, (), inverse_map)


def test_invoke_style2_jvp_rule():
    def dummy_jvp(x_tangent, x):
        return "x_tangent + 1"

    import inspect

    sig = inspect.signature(dummy_jvp)
    graph = LogicalGraph()
    node = LogicalNode("n", "Add", inputs=["i"])

    with mock.patch("ml_switcheroo_compiler.transforms.autodiff._compile_jvp_expr", return_value="compiled"):
        res = _invoke_style2_jvp_rule(dummy_jvp, sig, graph, node, ["t_i"])
        assert res == "compiled"

    # Return non-str
    def dummy_jvp2(x_tangent, x):
        return 42

    res2 = _invoke_style2_jvp_rule(dummy_jvp2, sig, graph, node, ["t_i"])
    assert res2 == 42

    # Error
    def dummy_jvp3(x_tangent, x):
        raise ValueError("err")

    with pytest.raises(MissingJVPRuleError):
        _invoke_style2_jvp_rule(dummy_jvp3, sig, graph, node, ["t_i"])


def test_invoke_jvp_rule():
    def jvp_style1(graph, node, tangent):
        return "t1"

    graph = LogicalGraph()
    node = LogicalNode("n", "Add", inputs=["i"])
    assert _invoke_jvp_rule(jvp_style1, graph, node, ["t"]) == "t1"

    def jvp_style1_err(graph, node, tangent):
        raise ValueError()

    with pytest.raises(MissingJVPRuleError):
        _invoke_jvp_rule(jvp_style1_err, graph, node, ["t"])

    def jvp_style2(tangent_x, x):
        return "t2"

    with mock.patch("ml_switcheroo_compiler.transforms.autodiff._invoke_style2_jvp_rule", return_value="t2"):
        assert _invoke_jvp_rule(jvp_style2, graph, node, ["t"]) == "t2"

    def jvp_bad(x):
        pass

    with pytest.raises(MissingJVPRuleError):
        _invoke_jvp_rule(jvp_bad, graph, node, ["t"])


def test_process_jvp_node():
    graph = LogicalGraph()

    # Test Output node
    n1 = LogicalNode("n1", "Output", inputs=["i1"], shape_metadata=())
    graph.nodes["i1"] = LogicalNode("i1", "Input", shape_metadata=())
    tangents = {"i1": "t1"}
    _process_jvp_node(graph, n1, tangents)
    assert "i1" in tangents
    assert "n1" in tangents

    n1_missing = LogicalNode("n1_missing", "Output", inputs=["i_missing"], shape_metadata=())
    graph.nodes["i_missing"] = LogicalNode("i_missing", "Input", shape_metadata=())
    tangents_empty = {}
    _process_jvp_node(graph, n1_missing, tangents_empty)
    assert "i_missing" in tangents_empty

    # Test Input/Constant (ignored)
    n2 = LogicalNode("n2", "Input")
    tangents2 = {}
    _process_jvp_node(graph, n2, tangents2)
    assert "n2" not in tangents2

    # Test tangent not present (ignored)
    n3 = LogicalNode("n3", "Add", inputs=["missing"])
    _process_jvp_node(graph, n3, tangents2)
    assert "n3" not in tangents2

    # Test normal node
    n4 = LogicalNode("n4", "Add", inputs=["i1"])
    tangents2["i1"] = "t1"
    with mock.patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp") as mock_get:
        mock_get.return_value = lambda graph, node, tangent: "t_out"
        _process_jvp_node(graph, n4, tangents2)
        assert tangents2["n4"] == "t_out"

        # Test missing JVP
        mock_get.side_effect = ValueError()
        with pytest.raises(ValueError):
            _process_jvp_node(graph, n4, tangents2)

        mock_get.side_effect = None
        with mock.patch("ml_switcheroo_compiler.transforms.autodiff._invoke_jvp_rule", side_effect=ValueError):
            with pytest.raises(ValueError):
                _process_jvp_node(graph, n4, tangents2)


def test_forward_pass_jvp():
    g = LogicalGraph()
    g.nodes["n1"] = LogicalNode("n1", "Input")
    _forward_pass_jvp(g, [g.nodes["n1"]], {})


def test_jvp_hvp():
    graph = LogicalGraph()
    graph.nodes["n1"] = LogicalNode("n1", "Input")
    graph.nodes["n2"] = LogicalNode("n2", "Output", shape_metadata=())

    with pytest.raises(ValueError):
        jvp(graph, ["p1"], ["t1", "t2"], ["n2"])

    with pytest.raises(ValueError):
        jvp(graph, ["p1"], ["t1"], ["missing"])

    with mock.patch("ml_switcheroo_compiler.transforms.autodiff._forward_pass_jvp"):
        g_out = jvp(graph, ["n1"], ["t1"], ["n2"])
        g_out2 = jvp(graph, ["n2"], ["t2"], ["n2"])
        assert len(g_out.outputs) == 1

    with mock.patch("ml_switcheroo_compiler.transforms.autodiff.grad") as mock_grad, mock.patch("ml_switcheroo_compiler.transforms.autodiff.jvp") as mock_jvp:
        mock_grad_out = LogicalGraph()
        mock_grad_out.outputs = ["grad_out"]
        mock_grad.return_value = mock_grad_out

        mock_jvp_out = LogicalGraph()
        mock_jvp_out.outputs = ["jvp_out"]
        mock_jvp.return_value = mock_jvp_out

        res = hvp(graph, ["p1"], ["t1"], ["out1"], mode="forward-over-reverse")
        assert res is mock_jvp_out

        res2 = hvp(graph, ["p1"], ["t1"], ["out1"], mode="reverse-over-forward")
        assert res2 is mock_grad_out

        with pytest.raises(ValueError):
            hvp(graph, ["p1"], ["t1"], ["out1", "out2"], mode="forward-over-reverse")

        with pytest.raises(ValueError):
            hvp(graph, ["p1"], ["t1"], ["out1"], mode="bad_mode")

        mock_jvp_out.outputs = ["out1", "out2"]
        with pytest.raises(ValueError):
            hvp(graph, ["p1"], ["t1"], ["out1"], mode="reverse-over-forward")


def test_autodiff_remaining_coverage_branches():
    """Test lines 27, 30, 32, 48, 50, 93->92, 115-117, 137, 309-317, and 582 in autodiff.py."""
    from ml_switcheroo_compiler.transforms.autodiff import (
        _add_nodes,
        _get_reachable_from_output,
        _is_zero_node,
        _load_rematerialization_rules,
        _process_jvp_node,
        _should_rematerialize,
        grad,
    )

    g_zero = LogicalGraph("g_zero")
    # 27: node_id not in graph.nodes
    assert _is_zero_node(g_zero, "nonexistent") is False

    # 30: Zeros / ZerosLike op_type
    n_zeros = LogicalNode("zeros_1", "Zeros", shape_metadata=())
    g_zero.nodes["zeros_1"] = n_zeros
    assert _is_zero_node(g_zero, "zeros_1") is True

    # 32: Constant 0.0
    n_const_0 = LogicalNode("const_0", "Constant", attributes={"value": 0.0}, shape_metadata=())
    g_zero.nodes["const_0"] = n_const_0
    assert _is_zero_node(g_zero, "const_0") is True

    # 48 & 50: _add_nodes with zero node
    n_other = LogicalNode("n_other", "Input", shape_metadata=())
    g_zero.nodes["n_other"] = n_other
    assert _add_nodes(g_zero, "zeros_1", "n_other") == "n_other"
    assert _add_nodes(g_zero, "n_other", "const_0") == "n_other"

    # 93->92: _get_reachable_from_output with unreachable node
    n_unreachable = LogicalNode("unreachable", "Input")
    n_out = LogicalNode("out_node", "Input")
    reachable = _get_reachable_from_output([n_unreachable, n_out], "out_node")
    assert "unreachable" not in reachable

    # 115-116: _load_rematerialization_rules exception during open
    with mock.patch("os.path.exists", return_value=True):
        with mock.patch("builtins.open", side_effect=PermissionError("mocked read error")):
            assert _load_rematerialization_rules() == {}

    # 117: yaml file does not exist
    with mock.patch("os.path.exists", return_value=False):
        assert _load_rematerialization_rules() == {}

    # 135: rules is empty dict
    node_chk = LogicalNode("node_chk", "Relu")
    assert _should_rematerialize(node_chk, rules={}) is False

    # 137: node in high_cost_ops
    node_high = LogicalNode("node_high", "Conv2D")
    rules_high = {"high_cost_ops": ["Conv2D"], "target_ops": ["Conv2D"]}
    assert _should_rematerialize(node_high, rules=rules_high) is False

    # 309-317: Vmap node vectorization lowering in grad
    g_vmap = LogicalGraph("g_vmap")
    g_vmap.inputs = ["in1"]
    g_vmap.outputs = ["v1"]
    g_vmap.nodes["in1"] = LogicalNode("in1", "Input", shape_metadata=(2, 4))
    g_vmap.nodes["v1"] = LogicalNode("v1", "Vmap", inputs=["in1"], shape_metadata=(2, 4))

    g_lowered = LogicalGraph("g_lowered")
    g_lowered.inputs = ["in1"]
    g_lowered.outputs = ["v1"]
    g_lowered.nodes["in1"] = LogicalNode("in1", "Input", shape_metadata=(2, 4))
    g_lowered.nodes["v1"] = LogicalNode("v1", "Identity", inputs=["in1"], shape_metadata=(2, 4))

    with mock.patch("ml_switcheroo_compiler.transforms.passes.vectorization.vectorization_pass", return_value=g_lowered):
        with mock.patch("ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry.get_vjp", return_value=lambda g, n, c: [c]):
            out_g = grad(g_vmap, ["in1"], "v1")
            assert out_g is not None

    # 582->exit: JVP rule returns None
    g_jvp = LogicalGraph("g_jvp")
    node_inp = LogicalNode("inp", "Input", shape_metadata=())
    g_jvp.nodes["inp"] = node_inp
    node_none = LogicalNode("n_none", "CustomOp", inputs=["inp"], shape_metadata=())
    g_jvp.nodes["n_none"] = node_none
    tangents = {"inp": "inp_tan"}
    with mock.patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp", return_value=lambda graph, node, tangents: None):
        _process_jvp_node(g_jvp, node_none, tangents)
        assert "n_none" not in tangents


def test_autodiff_extended_vjp_and_hvp_coverage():
    """Test cotangent_id variations, projected_tangents in hvp, and neural network VJP rules."""
    from ml_switcheroo_compiler.transforms.autodiff import (
        UnconnectedGradients,
        _avg_pool2d_vjp,
        _batch_norm_vjp,
        _conv2d_vjp_rule,
        _group_norm_vjp,
        _layer_norm_vjp,
        _max_pool2d_with_argmax_vjp,
        _rms_norm_vjp,
        conv2d_vjp_input,
        conv2d_vjp_weight,
        grad,
        hvp,
    )

    # cotangent_id as list, tuple, and scalar for multi-output grad
    g_multi = LogicalGraph("g_multi")
    g_multi.inputs = ["x"]
    g_multi.outputs = ["out1", "out2"]
    g_multi.nodes["x"] = LogicalNode("x", "Input", shape_metadata=(2,))
    g_multi.nodes["out1"] = LogicalNode("out1", "Identity", inputs=["x"], shape_metadata=(2,))
    g_multi.nodes["out2"] = LogicalNode("out2", "Identity", inputs=["x"], shape_metadata=(2,))
    g_multi.nodes["c1"] = LogicalNode("c1", "Input", shape_metadata=(2,))
    g_multi.nodes["c2"] = LogicalNode("c2", "Input", shape_metadata=(2,))

    with mock.patch("ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry.get_vjp", return_value=lambda g, n, c: [c]):
        out_list = grad(g_multi, ["x"], ["out1", "out2"], cotangent_id=["c1", "c2"])
        assert out_list is not None

        out_tuple = grad(g_multi, ["x"], ["out1", "out2"], cotangent_id=("c1", "c2"))
        assert out_tuple is not None

        out_single_for_multi = grad(g_multi, ["x"], ["out1", "out2"], cotangent_id="c1")
        assert out_single_for_multi is not None

    # hvp forward-over-reverse and reverse-over-forward with projected_tangents
    g_hvp = LogicalGraph("g_hvp")
    g_hvp.inputs = ["x"]
    g_hvp.outputs = ["y1", "y2"]
    g_hvp.nodes["x"] = LogicalNode("x", "Input", shape_metadata=(2,))
    g_hvp.nodes["y1"] = LogicalNode("y1", "Identity", inputs=["x"], shape_metadata=(2,))
    g_hvp.nodes["y2"] = LogicalNode("y2", "Identity", inputs=["x"], shape_metadata=(2,))

    with mock.patch("ml_switcheroo_compiler.transforms.autodiff.grad") as mock_g, mock.patch("ml_switcheroo_compiler.transforms.autodiff.jvp") as mock_j:
        m_graph = LogicalGraph("mock_graph")
        m_graph.outputs = ["y1", "y2"]
        mock_g.return_value = m_graph
        mock_j.return_value = m_graph

        hvp_for = hvp(g_hvp, ["x"], ["vx"], ["y1", "y2"], mode="forward-over-reverse", projected_tangents=["pt1", "pt2"])
        assert hvp_for is m_graph

        hvp_rof = hvp(g_hvp, ["x"], ["vx"], ["y1", "y2"], mode="reverse-over-forward", projected_tangents=["pt1", "pt2"])
        assert hvp_rof is m_graph

    # Neural network VJP rules
    g_nn = LogicalGraph("g_nn")
    g_nn.nodes["x"] = LogicalNode("x", "Input", shape_metadata=(1, 3, 16, 16))
    g_nn.nodes["w"] = LogicalNode("w", "Input", shape_metadata=(8, 3, 3, 3))
    g_nn.nodes["b"] = LogicalNode("b", "Input", shape_metadata=(8,))
    g_nn.nodes["cot"] = LogicalNode("cot", "Input", shape_metadata=(1, 8, 16, 16))

    # Conv2D VJP with strides tuple, dilations tuple
    conv_node_2 = LogicalNode(
        "conv_node_2",
        "Conv2D",
        inputs=["x", "w"],
        shape_metadata=(1, 8, 16, 16),
        attributes={"strides": (2, 2), "dilations": (1, 1), "padding": "SAME", "groups": 1},
    )
    in_adj = conv2d_vjp_input(g_nn, conv_node_2, "cot")
    w_adj = conv2d_vjp_weight(g_nn, conv_node_2, "cot")
    assert g_nn.nodes[in_adj].op_type == "Conv2DInputGrad"
    assert g_nn.nodes[w_adj].op_type == "Conv2DWeightGrad"

    res_rule_2 = _conv2d_vjp_rule(g_nn, conv_node_2, "cot")
    assert len(res_rule_2) == 2
    assert g_nn.nodes[res_rule_2[0]].op_type == "Conv2DInputGrad"
    assert g_nn.nodes[res_rule_2[1]].op_type == "Conv2DWeightGrad"

    # Conv2D VJP with 3 inputs (bias) and stride/dilation as keys
    conv_node_3 = LogicalNode(
        "conv_node_3",
        "Conv2D",
        inputs=["x", "w", "b"],
        shape_metadata=(1, 8, 16, 16),
        attributes={"stride": (1, 1), "dilation": (1, 1), "padding": "VALID", "groups": 1},
    )
    res_rule_3 = _conv2d_vjp_rule(g_nn, conv_node_3, "cot")
    assert len(res_rule_3) == 3
    assert g_nn.nodes[res_rule_3[0]].op_type == "Conv2DInputGrad"
    assert g_nn.nodes[res_rule_3[1]].op_type == "Conv2DWeightGrad"
    assert g_nn.nodes[res_rule_3[2]].op_type == "Conv2DBiasGrad"

    # MaxPool2DWithArgmax VJP (1 input and 2 inputs)
    mp_1 = LogicalNode("mp_1", "MaxPool2DWithArgmax", inputs=["x"], shape_metadata=(1, 3, 8, 8), attributes={})
    res_mp_1 = _max_pool2d_with_argmax_vjp(g_nn, mp_1, "cot")
    assert len(res_mp_1) == 1

    mp_2 = LogicalNode("mp_2", "MaxPool2DWithArgmax", inputs=["x", "w"], shape_metadata=(1, 3, 8, 8), attributes={})
    res_mp_2 = _max_pool2d_with_argmax_vjp(g_nn, mp_2, "cot")
    assert len(res_mp_2) == 2
    assert res_mp_2[1] == UnconnectedGradients.NONE

    # AvgPool2D VJP
    ap_node = LogicalNode("ap_node", "AvgPool2D", inputs=["x"], shape_metadata=(1, 3, 8, 8), attributes={})
    res_ap = _avg_pool2d_vjp(g_nn, ap_node, "cot")
    assert len(res_ap) == 1

    # BatchNorm VJP with 1, 2, 3, and 4 inputs
    g_nn.nodes["gamma"] = LogicalNode("gamma", "Input", shape_metadata=(3,))
    g_nn.nodes["beta"] = LogicalNode("beta", "Input", shape_metadata=(3,))
    g_nn.nodes["extra"] = LogicalNode("extra", "Input", shape_metadata=(3,))

    bn_1 = LogicalNode("bn_1", "BatchNorm", inputs=["x"], shape_metadata=(1, 3, 16, 16), attributes={})
    assert len(_batch_norm_vjp(g_nn, bn_1, "cot")) == 1

    bn_2 = LogicalNode("bn_2", "BatchNorm", inputs=["x", "gamma"], shape_metadata=(1, 3, 16, 16), attributes={})
    assert len(_batch_norm_vjp(g_nn, bn_2, "cot")) == 2

    bn_3 = LogicalNode("bn_3", "BatchNorm", inputs=["x", "gamma", "beta"], shape_metadata=(1, 3, 16, 16), attributes={})
    assert len(_batch_norm_vjp(g_nn, bn_3, "cot")) == 3

    bn_4 = LogicalNode("bn_4", "BatchNorm", inputs=["x", "gamma", "beta", "extra"], shape_metadata=(1, 3, 16, 16), attributes={})
    res_bn_4 = _batch_norm_vjp(g_nn, bn_4, "cot")
    assert len(res_bn_4) == 4
    assert res_bn_4[3] == UnconnectedGradients.NONE

    # LayerNorm VJP with 1, 2, 3, and 4 inputs
    ln_1 = LogicalNode("ln_1", "LayerNorm", inputs=["x"], shape_metadata=(1, 3, 16, 16), attributes={})
    assert len(_layer_norm_vjp(g_nn, ln_1, "cot")) == 1

    ln_2 = LogicalNode("ln_2", "LayerNorm", inputs=["x", "gamma"], shape_metadata=(1, 3, 16, 16), attributes={})
    assert len(_layer_norm_vjp(g_nn, ln_2, "cot")) == 2

    ln_3 = LogicalNode("ln_3", "LayerNorm", inputs=["x", "gamma", "beta"], shape_metadata=(1, 3, 16, 16), attributes={})
    assert len(_layer_norm_vjp(g_nn, ln_3, "cot")) == 3

    ln_4 = LogicalNode("ln_4", "LayerNorm", inputs=["x", "gamma", "beta", "extra"], shape_metadata=(1, 3, 16, 16), attributes={})
    res_ln_4 = _layer_norm_vjp(g_nn, ln_4, "cot")
    assert len(res_ln_4) == 4
    assert res_ln_4[3] == UnconnectedGradients.NONE

    # RMSNorm VJP with 1, 2, and 3 inputs
    rms_1 = LogicalNode("rms_1", "RMSNorm", inputs=["x"], shape_metadata=(1, 3, 16, 16), attributes={})
    assert len(_rms_norm_vjp(g_nn, rms_1, "cot")) == 1

    rms_2 = LogicalNode("rms_2", "RMSNorm", inputs=["x", "gamma"], shape_metadata=(1, 3, 16, 16), attributes={})
    assert len(_rms_norm_vjp(g_nn, rms_2, "cot")) == 2

    rms_3 = LogicalNode("rms_3", "RMSNorm", inputs=["x", "gamma", "extra"], shape_metadata=(1, 3, 16, 16), attributes={})
    res_rms_3 = _rms_norm_vjp(g_nn, rms_3, "cot")
    assert len(res_rms_3) == 3
    assert res_rms_3[2] == UnconnectedGradients.NONE

    # GroupNorm VJP with 1, 2, 3, and 4 inputs
    gn_1 = LogicalNode("gn_1", "GroupNorm", inputs=["x"], shape_metadata=(1, 3, 16, 16), attributes={})
    assert len(_group_norm_vjp(g_nn, gn_1, "cot")) == 1

    gn_2 = LogicalNode("gn_2", "GroupNorm", inputs=["x", "gamma"], shape_metadata=(1, 3, 16, 16), attributes={})
    assert len(_group_norm_vjp(g_nn, gn_2, "cot")) == 2

    gn_3 = LogicalNode("gn_3", "GroupNorm", inputs=["x", "gamma", "beta"], shape_metadata=(1, 3, 16, 16), attributes={})
    assert len(_group_norm_vjp(g_nn, gn_3, "cot")) == 3

    gn_4 = LogicalNode("gn_4", "GroupNorm", inputs=["x", "gamma", "beta", "extra"], shape_metadata=(1, 3, 16, 16), attributes={})
    res_gn_4 = _group_norm_vjp(g_nn, gn_4, "cot")
    assert len(res_gn_4) == 4
    assert res_gn_4[3] == UnconnectedGradients.NONE
