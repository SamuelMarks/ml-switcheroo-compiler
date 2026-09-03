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
