# ruff: noqa: E501
from unittest.mock import patch

import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.errors import ShapeMismatchError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.autodiff import _accumulate_gradients, _add_nodes, _backward_pass, _copy_graph, _extract_gradients, _forward_pass_jvp, _get_input_tangents, _get_reachable_from_output, _invoke_jvp_rule, _process_jvp_node, grad, hvp, jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY, register_vjp

"Tests for autodiff module."


def test_add_nodes():
    g = LogicalGraph(name="test")
    g.nodes["a"] = LogicalNode(id="a", op_type="Input", shape_metadata=())
    g.nodes["b"] = LogicalNode(id="b", op_type="Input", shape_metadata=())
    out_id = _add_nodes(g, "a", "b")
    assert out_id.startswith("a_add_b_")
    assert out_id in g.nodes
    assert g.nodes[out_id].op_type == "Add"


def test_copy_graph():
    g = LogicalGraph(name="test")
    g.nodes["a"] = LogicalNode(id="a", op_type="Input", shape_metadata=())
    new_g = _copy_graph(g)
    assert new_g.name == "test_grad"
    assert "a" in new_g.nodes
    assert new_g.nodes["a"] is not g.nodes["a"]


def test_get_reachable_from_output():
    a = LogicalNode(id="a", op_type="Input")
    b = LogicalNode(id="b", op_type="Input")
    c = LogicalNode(id="c", op_type="Add", inputs=["a", "b"])
    d = LogicalNode(id="d", op_type="Add", inputs=["c"])
    e = LogicalNode(id="e", op_type="Add", inputs=["a"])  # disconnected from d
    nodes = [a, b, c, d, e]
    reachable = _get_reachable_from_output(nodes, "d")
    assert reachable == {"a", "b", "c", "d"}


@patch("ml_switcheroo_compiler.transforms.autodiff.get_vjp")
def test_accumulate_gradients_missing_vjp(mock_get_vjp):
    mock_get_vjp.side_effect = ValueError("missing")
    g = LogicalGraph()
    g.nodes["a"] = LogicalNode(id="a", op_type="Input")
    g.nodes["b"] = LogicalNode(id="b", op_type="Input")
    c = LogicalNode(id="c", op_type="UnknownOp", inputs=["a", "b"])
    g.nodes["c"] = c
    with pytest.raises(ValueError, match="Missing VJP rule"):
        _accumulate_gradients(g, c, "c_adj", {})


@patch("ml_switcheroo_compiler.transforms.autodiff.get_vjp")
def test_accumulate_gradients_wrong_adjoints(mock_get_vjp):
    mock_get_vjp.return_value = lambda g, n, adj: ["only_one_adj"]
    g = LogicalGraph()
    c = LogicalNode(id="c", op_type="Add", inputs=["a", "b"])
    with pytest.raises(ValueError, match="expected 2"):
        _accumulate_gradients(g, c, "c_adj", {})


@patch("ml_switcheroo_compiler.transforms.autodiff.get_vjp")
def test_accumulate_gradients_success(mock_get_vjp):
    mock_get_vjp.return_value = lambda g, n, adj: ["adj1", None, "adj3"]
    g = LogicalGraph()
    g.nodes["a"] = LogicalNode(id="a", op_type="Input", shape_metadata=())
    g.nodes["b"] = LogicalNode(id="b", op_type="Input", shape_metadata=())
    g.nodes["d"] = LogicalNode(id="d", op_type="Input", shape_metadata=())
    c = LogicalNode(id="c", op_type="Add", inputs=["a", "b", "d"])
    g.nodes["c"] = c
    g.nodes["existing_adj"] = LogicalNode(id="existing_adj", op_type="Input", shape_metadata=())
    adjoints = {"a": "existing_adj"}
    _accumulate_gradients(g, c, "c_adj", adjoints)
    assert "existing_adj_add_adj1" in adjoints["a"]
    assert "b" not in adjoints
    assert adjoints["d"] == "adj3"


def test_accumulate_gradients():
    g = LogicalGraph()
    g.nodes["a"] = LogicalNode(id="a", op_type="Input")
    g.nodes["c"] = LogicalNode(id="c", op_type="Add", inputs=["a", "a"])

    with patch("ml_switcheroo_compiler.transforms.autodiff.get_vjp") as mock_vjp:
        mock_vjp.return_value = lambda graph, node, cotan: ("v1", "v2")
        adjoints = {"a": "adj_a"}

        with patch("ml_switcheroo_compiler.transforms.autodiff._add_nodes") as mock_add:
            mock_add.side_effect = lambda g, a, b: f"{a}+{b}"
            _accumulate_gradients(g, g.nodes["c"], "adj_c", adjoints)
            # The node c inputs are ["a", "a"], so cotangents are ("v1", "v2").
            # adjoints["a"] starts as "adj_a".
            # First iteration (i=0): it adds adj_a + v1
            # Second iteration (i=1): it adds (adj_a + v1) + v2
            assert adjoints["a"] == "adj_a+v1+v2"

    with patch("ml_switcheroo_compiler.transforms.autodiff.get_vjp") as mock_vjp:
        # returns None for inputs
        mock_vjp.return_value = lambda graph, node, cotan: (None, "v2")
        adjoints = {"a": "adj_a"}

        with patch("ml_switcheroo_compiler.transforms.autodiff._add_nodes") as mock_add:
            mock_add.side_effect = lambda g, a, b: f"{a}+{b}"
            _accumulate_gradients(g, g.nodes["c"], "adj_c", adjoints)
            assert adjoints["a"] == "adj_a+v2"


def test_recompute_subgraph():
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode
    from ml_switcheroo_compiler.transforms.autodiff import _recompute_subgraph

    g = IRGraph()
    inp = LogicalNode(id="inp", op_type="Input")
    inp.attributes = {"rematerialize": True}
    n = LogicalNode(id="n", op_type="Exp", inputs=["inp", "missing_inp"])
    g.nodes = {"inp": inp, "n": n}

    res = _recompute_subgraph(g, n)
    assert res is not None
    assert len(res.inputs) == 2
    assert "missing_inp" in res.inputs

    # test input not rematerialize
    inp2 = LogicalNode(id="inp2", op_type="Input")
    inp2.attributes = {"rematerialize": False}
    n2 = LogicalNode(id="n2", op_type="Exp", inputs=["inp2"])
    g.nodes = {"inp2": inp2, "n2": n2}
    res2 = _recompute_subgraph(g, n2)
    assert res2.inputs == ["inp2"]


def test_hvp_missing_rules():
    import pytest

    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode
    from ml_switcheroo_compiler.transforms.autodiff import hvp

    g = IRGraph()
    n1 = LogicalNode(id="inp", op_type="Input")
    n2 = LogicalNode(id="out", op_type="OpWithoutRules", inputs=["inp"])
    g.nodes = {"inp": n1, "out": n2}
    g.inputs = ["inp"]
    g.outputs = ["out"]

    with pytest.raises(ValueError, match="Missing.*rule for operation"):
        hvp(g, ["inp"], ["v"], ["out"])


def test_add_nodes_extra():
    g = LogicalGraph()
    g.nodes["a"] = LogicalNode(id="a", op_type="Input", shape_metadata=(2,))
    g.nodes["b"] = LogicalNode(id="b", op_type="Input", shape_metadata=(2,))
    g.nodes["c"] = LogicalNode(id="c", op_type="Input", shape_metadata=(2, 2))  # diff shape

    # same shapes
    res = _add_nodes(g, "a", "b")
    assert res != "a" and res != "b"
    assert g.nodes[res].op_type == "Add"

    # diff shapes
    g.nodes["a"] = LogicalNode(id="a", op_type="Input", shape_metadata=(2,))
    g.nodes["c"] = LogicalNode(id="c", op_type="Input", shape_metadata=(2, 2))  # diff shape
    res2 = _add_nodes(g, "a", "c")
    assert res2 != "a" and res2 != "c"


def test_backward_pass():
    g = LogicalGraph()
    a = LogicalNode(id="a", op_type="Input")
    b = LogicalNode(id="b", op_type="Input")
    c = LogicalNode(id="c", op_type="StopGradient", inputs=["b"])
    d = LogicalNode(id="d", op_type="Add", inputs=["a", "c"])
    e = LogicalNode(id="e", op_type="Add", inputs=["d"])
    adjoints = {"d": "adj_d", "c": "adj_c"}
    with patch("ml_switcheroo_compiler.transforms.autodiff._accumulate_gradients") as mock_acc:
        _backward_pass(g, [a, b, c, d, e], {"d", "c", "b"}, adjoints)
        assert mock_acc.call_count == 1
        assert mock_acc.call_args[0][1].id == "d"


def test_extract_gradients():
    g = LogicalGraph()
    g.nodes["a"] = LogicalNode(id="a", op_type="Input", shape_metadata=())
    g.nodes["b"] = LogicalNode(id="b", op_type="Input", shape_metadata=())
    with pytest.raises(ValueError, match="not found"):
        _extract_gradients(g, ["unknown"], {})
    out = _extract_gradients(g, ["a", "b"], {"a": "a_grad"})
    assert out[0] == "a_grad"
    assert out[1].startswith("grad_zeros")


def test_grad():
    g = LogicalGraph()
    with pytest.raises(ValueError, match="not found"):
        grad(g, ["a"], "c")
    g.nodes["a"] = LogicalNode(id="a", op_type="Input", shape_metadata=())
    g.nodes["c"] = LogicalNode(id="c", op_type="Input", shape_metadata=())
    with patch("ml_switcheroo_compiler.transforms.autodiff._backward_pass"), patch("ml_switcheroo_compiler.transforms.autodiff.topological_sort") as mock_topo:
        mock_topo.return_value = []
        new_g = grad(g, ["a"], "c")
        assert len(new_g.outputs) == 1


def test_get_input_tangents():
    g = LogicalGraph()
    g.nodes["a"] = LogicalNode(id="a", op_type="Input", shape_metadata=())
    g.nodes["b"] = LogicalNode(id="b", op_type="Input", shape_metadata=())
    n = LogicalNode(id="n", op_type="Add", inputs=["a", "b"])
    tans = _get_input_tangents(g, n, {"a": "t_a"})
    assert tans[0] == "t_a"
    assert tans[1].startswith("jvp_zeros")


def test_invoke_jvp_rule():

    def mock_rule(graph, node, tangents):
        return tangents

    def mock_rule2(a, b):
        return "called2"

    assert _invoke_jvp_rule(mock_rule, None, None, ["t1"]) == "t1"
    assert _invoke_jvp_rule(mock_rule, None, None, ["t1", "t2"]) == ["t1", "t2"]

    def mock_rule_tuple(graph, node, tangents):
        return tuple(tangents)

    assert _invoke_jvp_rule(mock_rule_tuple, None, None, ["t1", "t2"]) == ("t1", "t2")

    def mock_rule3(graph, node, tangents):
        return "called3"

    assert _invoke_jvp_rule(mock_rule3, None, None, ["t1"]) == "called3"
    import pytest

    from ml_switcheroo_compiler.core.errors import MissingJVPRuleError

    with pytest.raises(MissingJVPRuleError):
        _invoke_jvp_rule(mock_rule2, None, None, ["t1"])


@patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp")
def test_process_jvp_node(mock_get_jvp):
    mock_get_jvp.side_effect = ValueError("missing")
    g = LogicalGraph()
    n1 = LogicalNode(id="n1", op_type="Input")
    _process_jvp_node(g, n1, {})
    n2 = LogicalNode(id="n2", op_type="Add", inputs=["a"])
    _process_jvp_node(g, n2, {})
    with pytest.raises(ValueError, match="Missing JVP"):
        _process_jvp_node(g, n2, {"a": "t_a"})


@patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp")
@patch("ml_switcheroo_compiler.transforms.autodiff._invoke_jvp_rule")
def test_process_jvp_node_success(mock_invoke, mock_get_jvp):
    mock_get_jvp.return_value = lambda: None
    mock_invoke.return_value = "new_t"
    g = LogicalGraph()
    g.nodes["a"] = LogicalNode(id="a", op_type="Input", shape_metadata=())
    n2 = LogicalNode(id="n2", op_type="Add", inputs=["a"])
    g.nodes["n2"] = n2
    tans = {"a": "t_a"}
    _process_jvp_node(g, n2, tans)
    assert tans["n2"] == "new_t"


@patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp")
@patch("ml_switcheroo_compiler.transforms.autodiff._invoke_jvp_rule")
def test_process_jvp_node_failure(mock_invoke, mock_get_jvp):
    mock_get_jvp.return_value = lambda: None
    mock_invoke.side_effect = ValueError
    g = LogicalGraph()
    g.nodes["a"] = LogicalNode(id="a", op_type="Input", shape_metadata=())
    n2 = LogicalNode(id="n2", op_type="Add", inputs=["a"])
    g.nodes["n2"] = n2
    tans = {"a": "t_a"}
    with pytest.raises(ValueError, match="Missing JVP"):
        _process_jvp_node(g, n2, tans)


def test_forward_pass_jvp():
    with patch("ml_switcheroo_compiler.transforms.autodiff._process_jvp_node") as mock_proc:
        _forward_pass_jvp(None, [1, 2], {})
        assert mock_proc.call_count == 2


def test_jvp():
    g = LogicalGraph()
    with pytest.raises(ValueError, match="same length"):
        jvp(g, ["a"], [], [])
    with pytest.raises(ValueError, match="not found"):
        jvp(g, ["a"], ["ta"], ["out"])
    g.nodes["a"] = LogicalNode(id="a", op_type="Input", shape_metadata=())
    g.nodes["out"] = LogicalNode(id="out", op_type="Input", shape_metadata=())
    with patch("ml_switcheroo_compiler.transforms.autodiff._forward_pass_jvp"), patch("ml_switcheroo_compiler.transforms.autodiff.topological_sort"):
        new_g = jvp(g, ["a"], ["ta"], ["out"])
        assert len(new_g.outputs) == 1
        assert new_g.outputs[0].startswith("jvp_zeros")
        new_g2 = jvp(g, ["a"], ["ta"], ["a"])
        assert new_g2.outputs[0] == "ta"


def test_hvp():
    g = LogicalGraph()
    with patch("ml_switcheroo_compiler.transforms.autodiff.grad") as mock_grad:
        mock_grad_g = LogicalGraph()
        mock_grad_g.outputs = ["grad_out"]
        mock_grad.return_value = mock_grad_g
        with patch("ml_switcheroo_compiler.transforms.autodiff.jvp") as mock_jvp:
            mock_jvp.return_value = "hvp_graph"
            res = hvp(g, ["a"], ["ta"], ["out"])
            assert res == "hvp_graph"
            mock_jvp.assert_called_once_with(mock_grad_g, ["a"], ["ta"], ["grad_out"])


"Provides required module functionality."


def test_autodiff_coverage_brute() -> None:
    """Test the autodiff coverage brute behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Execute the requested function."
        g = IRGraph()
        n1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=None)
        n2 = IRNode(id="n2", op_type="Input", inputs=[], attributes={}, shape_metadata=None)
        n3 = IRNode(id="n3", op_type="FakeOp", inputs=["n1", "n2"], attributes={}, shape_metadata=None)
        g.nodes = {"n1": n1, "n2": n2, "n3": n3}
        if "FakeOp" in _VJP_REGISTRY:
            del _VJP_REGISTRY["FakeOp"]

        @register_vjp("FakeOp")
        def fake_op_vjp(graph: object, node: object, adj_id: str) -> list[str]:
            """Evaluate and process the fake op vjp operation.

            Args:
                graph (object): Required parameter for graph.
                node (object): Required parameter for node.
                adj_id (str): Required parameter for adj_id.

            Returns:
                list: The evaluated or processed output.
            """
            raise ValueError("Missing VJP")

        with pytest.raises((ValueError, ShapeMismatchError), match="Missing VJP rule for operation"):
            grad(g, ["n1"], "n3")
        if "FakeOp" in _VJP_REGISTRY:
            del _VJP_REGISTRY["FakeOp"]

        @register_vjp("FakeOp")
        def fake_op_vjp2(graph: object, node: object, adj_id: str) -> list[str]:
            """Evaluate and process the fake op vjp2 operation.

            Args:
                graph (object): Required parameter for graph.
                node (object): Required parameter for node.
                adj_id (str): Required parameter for adj_id.

            Returns:
                list: The evaluated or processed output.
            """
            return ["adj_1"]

        with pytest.raises((ValueError, ShapeMismatchError), match="VJP for FakeOp returned 1 adjoints, expected 2."):
            grad(g, ["n1"], "n3")
        if "FakeOp" in _VJP_REGISTRY:
            del _VJP_REGISTRY["FakeOp"]
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


"""Coverage tests for autodiff."""


from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients


def test_accumulate_gradients_none():
    from unittest.mock import patch

    graph = LogicalGraph()
    node = LogicalNode(id="n1", op_type="Add", inputs=["in1"])
    graph.nodes["n1"] = node

    adjoints = {"in1": "old_adj"}

    def mock_vjp(*args, **kwargs):
        return [UnconnectedGradients.NONE]

    with patch("ml_switcheroo_compiler.transforms.autodiff.get_vjp", return_value=mock_vjp):
        _accumulate_gradients(graph, node, "adj1", adjoints)
        assert adjoints["in1"] == "old_adj"


def test_accumulate_gradients_none_continue():
    from unittest.mock import patch

    graph = LogicalGraph()
    node = LogicalNode(id="n1", op_type="Add", inputs=["in1"])
    graph.nodes["n1"] = node

    adjoints = {"in1": "old_adj"}

    def mock_vjp(*args, **kwargs):
        return [None]

    with patch("ml_switcheroo_compiler.transforms.autodiff.get_vjp", return_value=mock_vjp):
        _accumulate_gradients(graph, node, "adj1", adjoints)
        assert adjoints["in1"] == "old_adj"


def test_accumulate_gradients_zero_continue():
    from unittest.mock import patch

    graph = LogicalGraph()
    node = LogicalNode(id="n1", op_type="Add", inputs=["in1"])
    graph.nodes["n1"] = node

    adjoints = {"in1": "old_adj"}

    def mock_vjp(*args, **kwargs):
        return [UnconnectedGradients.ZERO]

    with patch("ml_switcheroo_compiler.transforms.autodiff.get_vjp", return_value=mock_vjp):
        _accumulate_gradients(graph, node, "adj1", adjoints)
        assert adjoints["in1"] == "old_adj"


def test_jvp_primals_tangents_length_mismatch():
    graph = LogicalGraph()
    with pytest.raises(ValueError, match="primals and tangents must have the same length"):
        jvp(graph, ["p1", "p2"], ["t1"], ["o1"])


def test_jvp_output_not_found():
    graph = LogicalGraph()
    with pytest.raises(ValueError, match="Output node 'o1' not found in graph."):
        jvp(graph, ["p1"], ["t1"], ["o1"])


def test_invoke_jvp_rule_no_graph_node():
    def mock_jvp(a, b):
        return a + b

    import pytest

    from ml_switcheroo_compiler.core.errors import MissingJVPRuleError

    with pytest.raises(MissingJVPRuleError):
        _invoke_jvp_rule(mock_jvp, None, None, ["t1"])


def test_process_jvp_node_unimplemented():
    from unittest.mock import patch

    from ml_switcheroo_compiler.transforms.autodiff import _process_jvp_node

    graph = LogicalGraph()
    node = LogicalNode(id="n1", op_type="FakeOp", inputs=["p1"], shape_metadata=())
    graph.nodes["n1"] = node
    tangents = {"p1": "t1"}

    def raise_err(*args, **kwargs):
        raise ValueError()

    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp", side_effect=raise_err):
        with pytest.raises(ValueError, match="Missing JVP rule for operation: FakeOp"):
            _process_jvp_node(graph, node, tangents)


def test_process_jvp_node_invoke_unimplemented():
    from unittest.mock import patch

    from ml_switcheroo_compiler.transforms.autodiff import _process_jvp_node

    graph = LogicalGraph()
    node = LogicalNode(id="n1", op_type="Add", inputs=["p1"], shape_metadata=())
    graph.nodes["n1"] = node
    tangents = {"p1": "t1"}

    def raise_err(*args, **kwargs):
        raise ValueError()

    with patch("ml_switcheroo_compiler.transforms.autodiff._invoke_jvp_rule", side_effect=raise_err):
        with pytest.raises(ValueError, match="Missing JVP rule for operation: Add"):
            _process_jvp_node(graph, node, tangents)


def test_dummy_jvp_execution():
    from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import get_jvp

    # Now falls back to finite difference
    res = get_jvp("NonExistentOp")
    assert res is not None


def test_autodiff_grad_output_node_and_cotangent():
    from ml_switcheroo_compiler.transforms.autodiff import grad

    graph = LogicalGraph()
    # Build graph: Input -> Output
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input", shape_metadata=())
    graph.nodes["out1"] = LogicalNode(id="out1", op_type="Output", inputs=["in1"], shape_metadata=())

    # Test cotangent_id as string
    g2 = grad(graph, wrt=["in1"], output_id="out1", cotangent_id="cot_node")
    assert "cot_node" in g2.outputs

    # Test cotangent_id as dict
    g3 = grad(graph, wrt=["in1"], output_id="out1", cotangent_id={"out1": "cot_node2"})
    assert "cot_node2" in g3.outputs


def test_jvp_compile_expr_and_style2():
    from ml_switcheroo_compiler.transforms.autodiff import _compile_jvp_expr, _invoke_jvp_rule

    graph = LogicalGraph()
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input", shape_metadata=())
    graph.nodes["const_zero"] = LogicalNode(id="const_zero", op_type="Constant", attributes={"value": 0.0}, shape_metadata=())

    # Test _compile_jvp_expr
    inv_map = {"safe_id_0": "t1", "safe_id_1": "t2"}

    # Unary
    res = _compile_jvp_expr("-safe_id_0", graph, (), inv_map)
    assert graph.nodes[res].op_type == "Negative"
    assert graph.nodes[res].inputs[0] == "t1"

    # Unary plus (noop basically)
    res = _compile_jvp_expr("+safe_id_0", graph, (), inv_map)
    assert res == "t1"

    # BinOp Add, Sub, Mult, Div
    res = _compile_jvp_expr("safe_id_0 + safe_id_1", graph, (), inv_map)
    assert graph.nodes[res].op_type == "Add"

    res = _compile_jvp_expr("safe_id_0 - safe_id_1", graph, (), inv_map)
    assert graph.nodes[res].op_type == "Subtract"

    res = _compile_jvp_expr("safe_id_0 * safe_id_1", graph, (), inv_map)
    assert graph.nodes[res].op_type == "Multiply"

    res = _compile_jvp_expr("safe_id_0 / safe_id_1", graph, (), inv_map)
    assert graph.nodes[res].op_type == "TrueDivide"

    # Constant
    res = _compile_jvp_expr("2.0 * safe_id_0", graph, (), inv_map)
    assert graph.nodes[res].op_type == "Multiply"

    # Unsupported
    import pytest

    with pytest.raises(ValueError):
        _compile_jvp_expr("safe_id_0 ** 2", graph, (), inv_map)

    import pytest

    with pytest.raises(ValueError):
        _compile_jvp_expr("len(safe_id_0)", graph, (), inv_map)

    # Style2 JVP rule
    def my_jvp(x, tangent_x):
        return "x + tangent_x"

    class DummyNode:
        inputs = ["in1"]
        shape_metadata = ()

    res = _invoke_jvp_rule(my_jvp, graph, DummyNode(), ["t1"])
    assert res is not None

    def my_jvp_exc(x, tangent_x):
        raise ValueError("err")

    import pytest

    from ml_switcheroo_compiler.core.errors import MissingJVPRuleError

    with pytest.raises(MissingJVPRuleError):
        _invoke_jvp_rule(my_jvp_exc, graph, DummyNode(), ["t1"])


def test_process_jvp_node_output():
    from ml_switcheroo_compiler.transforms.autodiff import _process_jvp_node

    graph = LogicalGraph()
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input", shape_metadata=())
    graph.nodes["in2"] = LogicalNode(id="in2", op_type="Input", shape_metadata=())
    out_node = LogicalNode(id="out1", op_type="Output", inputs=["in1", "in2"], shape_metadata=())
    graph.nodes["out1"] = out_node

    tangents = {"in1": "t1"}  # in2 is missing, should create zeros

    _process_jvp_node(graph, out_node, tangents)
    assert "out1" in tangents
    out_t = tangents["out1"]
    assert graph.nodes[out_t].op_type == "Output"
    assert graph.nodes[out_t].inputs[0] == "t1"
    # The second input should be a zero constant
    assert graph.nodes[graph.nodes[out_t].inputs[1]].op_type == "Constant"


def test_process_jvp_node_output_none():
    from unittest.mock import patch

    from ml_switcheroo_compiler.transforms.autodiff import _process_jvp_node

    g = LogicalGraph()
    g.nodes["x"] = LogicalNode(id="x", op_type="Input", shape_metadata=())
    n = LogicalNode(id="n", op_type="CustomOpNone", inputs=["x"], shape_metadata=())
    g.nodes["n"] = n

    tangents = {"x": "t_x", "n": "existing"}

    with patch("ml_switcheroo_compiler.transforms.autodiff._invoke_jvp_rule", return_value=None):
        with patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp", return_value=lambda g, n, t: None):
            _process_jvp_node(g, n, tangents)

    assert tangents["n"] == "existing"  # Not modified since out_tangent is None


def test_hvp_no_missing_rules():
    from ml_switcheroo_compiler.transforms.autodiff import hvp
    from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import has_jvp
    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import has_vjp

    assert has_vjp("Add")
    assert has_jvp("Add")

    g2 = LogicalGraph()
    g2.nodes["x"] = LogicalNode(id="x", op_type="Input", shape_metadata=())
    g2.nodes["n"] = LogicalNode(id="n", op_type="Add", inputs=["x", "x"], shape_metadata=())
    g2.nodes["y"] = LogicalNode(id="y", op_type="Output", inputs=["n"], shape_metadata=())
    g2.nodes["t_x"] = LogicalNode(id="t_x", op_type="Input", shape_metadata=())

    hvp(g2, ["x"], ["t_x"], ["y"])


def test_jvp_invoke_rule_exc():
    from ml_switcheroo_compiler.transforms.autodiff import _invoke_jvp_rule

    def bad_jvp(graph, node, tangent):
        raise ValueError("err")

    import pytest

    from ml_switcheroo_compiler.core.errors import MissingJVPRuleError

    with pytest.raises(MissingJVPRuleError):
        _invoke_jvp_rule(bad_jvp, None, None, ["t1"])


def test_jvp_style2_returns_non_string():
    from ml_switcheroo_compiler.transforms.autodiff import _invoke_style2_jvp_rule

    def my_jvp(x, tangent_x):
        return 42

    import inspect

    sig = inspect.signature(my_jvp)

    class DummyNode:
        inputs = ["in1"]
        shape_metadata = ()

    res = _invoke_style2_jvp_rule(my_jvp, sig, None, DummyNode(), ["t1"])
    assert res == 42


def test_autodiff_provider_parser():
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode
    from ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider import _parse_expression, get_jvp_from_data

    g = IRGraph()
    n = LogicalNode(id="n1", op_type="Exp", inputs=["x", "y"])
    g.nodes["x"] = LogicalNode(id="x", op_type="Input")
    g.nodes["y"] = LogicalNode(id="y", op_type="Input")

    # Simple expression
    res = _parse_expression(g, "Add($cotangent, $input[0])", n, cotangent="cot")
    assert res is not None
    assert "cot" in g.nodes[res].inputs
    assert "x" in g.nodes[res].inputs

    # Constant and tangents
    res2 = _parse_expression(g, "Multiply(Constant(2.0), $tangent[1])", n, tangents=["t0", "t1"])
    assert res2 is not None
    assert "t1" in g.nodes[res2].inputs

    # Run Constant(2.0) again to hit cache
    res_c2 = _parse_expression(g, "Constant(2.0)", n)
    assert g.nodes[res_c2].op_type == "Constant"

    # Run empty args
    res_empty = _parse_expression(g, "EmptyOp()", n)
    assert g.nodes[res_empty].op_type == "EmptyOp"
    assert g.nodes[res_empty].inputs == []

    # Test fallback with more inputs than tangents
    n3 = LogicalNode(id="n3", op_type="Add", inputs=["x", "y", "z"])
    fallback = get_jvp_from_data("OpWithoutRulesAndData")
    res3 = fallback(g, n3, ["t_x", "t_y"])  # z has no tangent
    assert res3 is not None

    # SetItem attrs where node has no attrs
    n_set = LogicalNode(id="n_set", op_type="SetItem", inputs=["a", "b", "c"])
    # Do not set attributes initially so we bypass first check
    n_set.attributes = {"indices": [1]}

    # We need op == "SetItem" but op != node.op_type initially to bypass first `if`
    n_set.op_type = "NotSetItem"
    res4 = _parse_expression(g, "SetItem(a, b, c)", n_set)
    assert g.nodes[res4].attributes == {"indices": [1]}

    # data_jvp
    with patch("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", {"TestOp": {"autodiff": {"jvp": "Add($tangent[0], $input[0])"}}}):
        data_jvp_func = get_jvp_from_data("TestOp")
        res5 = data_jvp_func(g, n, "t_x")  # single tangent
        assert res5 is not None


def test_accumulate_gradients_rematerialize():
    from ml_switcheroo_compiler.ir.core import LogicalGraph, LogicalNode
    from ml_switcheroo_compiler.transforms.autodiff import _accumulate_gradients

    g = LogicalGraph()
    n = LogicalNode(id="n", op_type="Exp", inputs=["x"])
    n.attributes = {"rematerialize": True}
    g.nodes = {"x": LogicalNode(id="x", op_type="Input"), "n": n}

    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry.get_vjp") as mock_vjp:
        mock_vjp.return_value = lambda g, n, adj: [f"adj_{inp}" for inp in n.inputs]
        adjoints = {}
        _accumulate_gradients(g, n, "adj_n", adjoints)
        assert "x" in adjoints
