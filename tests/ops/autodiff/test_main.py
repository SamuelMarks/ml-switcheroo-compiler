# ruff: noqa: E501
import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.errors import MissingJVPRuleError
from ml_switcheroo_compiler.transforms.autodiff import _accumulate_gradients, _add_nodes, _backward_pass, _copy_graph, _extract_gradients, _get_input_tangents, _get_reachable_from_output, _invoke_jvp_rule, _process_jvp_node, grad, hvp, jvp


def test_add_nodes():
    g = LogicalGraph()
    n1 = LogicalNode("n1", "Input", shape_metadata=(2,))
    n2 = LogicalNode("n2", "Input", shape_metadata=(2,))
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    out_id = _add_nodes(g, "n1", "n2")
    assert out_id in g.nodes
    assert g.nodes[out_id].op_type == "Add"


def test_copy_graph():
    g = LogicalGraph(name="test")
    n1 = LogicalNode("n1", "Input", shape_metadata=(2,))
    g.nodes["n1"] = n1
    g2 = _copy_graph(g)
    assert g2.name == "test_grad"
    assert "n1" in g2.nodes


def test_get_reachable_from_output():
    n1 = LogicalNode("n1", "Input", [])
    n2 = LogicalNode("n2", "Add", inputs=["n1", "n1"])
    n3 = LogicalNode("n3", "Mul", inputs=["n2", "n2"])
    assert _get_reachable_from_output([n1, n2, n3], "n3") == {"n3", "n2", "n1"}


def test_accumulate_gradients(mocker):
    g = LogicalGraph()
    n1 = LogicalNode("n1", "Mul", inputs=["a", "b"])
    mock_get_vjp = mocker.patch("ml_switcheroo_compiler.transforms.autodiff.get_vjp")

    def dummy_vjp(graph, node, adj):
        return ["da", "db"]

    mock_get_vjp.return_value = lambda g, n, adj: ["da", "db"]
    adj = {}
    _accumulate_gradients(g, n1, "adj", adj)
    assert adj == {"a": "da", "b": "db"}
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff._add_nodes", return_value="added")
    _accumulate_gradients(g, n1, "adj", adj)
    assert adj == {"a": "added", "b": "added"}
    mock_get_vjp.side_effect = ValueError
    with pytest.raises(ValueError):
        _accumulate_gradients(g, n1, "adj", adj)
    mock_get_vjp.side_effect = None
    mock_get_vjp.return_value = lambda g, n, adj: ["da"]
    with pytest.raises(ValueError):
        _accumulate_gradients(g, n1, "adj", adj)


def test_backward_pass(mocker):
    g = LogicalGraph()
    n1 = LogicalNode("n1", "Mul", inputs=["a", "b"])
    n2 = LogicalNode("n2", "Input", [])
    mock_acc = mocker.patch("ml_switcheroo_compiler.transforms.autodiff._accumulate_gradients")
    _backward_pass(g, [n1, n2], {"n1"}, {"n1": "adj"})
    mock_acc.assert_called_once()
    mock_acc.reset_mock()
    _backward_pass(g, [n1], set(), {"n1": "adj"})
    mock_acc.assert_not_called()
    mock_acc.reset_mock()
    _backward_pass(g, [n2], {"n2"}, {"n2": "adj"})
    mock_acc.assert_not_called()


def test_extract_gradients():
    g = LogicalGraph()
    n1 = LogicalNode("n1", "Input", shape_metadata=(2,))
    g.nodes["n1"] = n1
    assert _extract_gradients(g, ["n1"], {"n1": "adj"}) == ["adj"]
    res = _extract_gradients(g, ["n1"], {})
    assert len(res) == 1
    assert "grad_zeros" in res[0]
    with pytest.raises(ValueError):
        _extract_gradients(g, ["not_exist"], {})


def test_grad(mocker):
    g = LogicalGraph()
    n1 = LogicalNode("n1", "Input", shape_metadata=(2,))
    n2 = LogicalNode("n2", "Mul", inputs=["n1", "n1"])
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    mock_get_vjp = mocker.patch("ml_switcheroo_compiler.transforms.autodiff.get_vjp")

    def dummy_vjp(graph, node, adj):
        return ["da", "db"]

    mock_get_vjp.return_value = lambda g, n, adj: ["da", "db"]
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff._add_nodes", return_value="added")
    with pytest.raises(ValueError):
        grad(g, ["n1"], "not_exist")
    g2 = grad(g, ["n1"], "n2")
    assert g2 is not None


def test_get_input_tangents():
    g = LogicalGraph()
    n1 = LogicalNode("n1", "Input", shape_metadata=(2,))
    g.nodes["n1"] = n1
    n2 = LogicalNode("n2", "Mul", inputs=["n1", "n3"])
    g.nodes["n3"] = LogicalNode("n3", "Input", shape_metadata=())
    res = _get_input_tangents(g, n2, {"n1": "t1"})
    assert res[0] == "t1"
    assert "jvp_zeros" in res[1]


def test_invoke_jvp_rule():

    def mock_jvp(graph, node, tangent):
        return "res"

    assert _invoke_jvp_rule(mock_jvp, None, None, ["t"]) == "res"

    def mock_jvp_bad():
        pass

    import pytest

    with pytest.raises(MissingJVPRuleError):
        _invoke_jvp_rule(mock_jvp_bad, None, None, ["t"])


def test_process_jvp_node(mocker):
    g = LogicalGraph()
    n1 = LogicalNode("n1", "Mul", inputs=["a", "b"])
    mock_get_jvp = mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp")
    mock_get_jvp.return_value = lambda graph, node, tangents: "out_t"
    _process_jvp_node(g, n1, {})
    tangents = {"a": "t1"}
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff._get_input_tangents", return_value=["t1", "t2"])
    _process_jvp_node(g, n1, tangents)
    assert tangents["n1"] == "out_t"
    mock_get_jvp.side_effect = ValueError
    with pytest.raises(ValueError):
        _process_jvp_node(g, n1, tangents)


def test_jvp_hvp(mocker):
    g = LogicalGraph()
    n1 = LogicalNode("n1", "Input", shape_metadata=(2,))
    n2 = LogicalNode("n2", "Mul", inputs=["n1", "n1"])
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    mock_get_jvp = mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp")
    mock_get_jvp.return_value = lambda graph, node, tangents: "out_t"
    with pytest.raises(ValueError):
        jvp(g, ["n1"], ["t1", "t2"], ["n2"])
    with pytest.raises(ValueError):
        jvp(g, ["n1"], ["t1"], ["not_exist"])
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff._get_input_tangents", return_value=["t1", "t1"])
    g2 = jvp(g, ["n1"], ["t1"], ["n2"])
    assert g2 is not None
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff.grad", return_value=g2)
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry.has_vjp", return_value=True)
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.has_jvp", return_value=True)
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff.jvp", return_value="hvp_res")
    assert hvp(g, ["n1"], ["t1"], ["n2"]) == "hvp_res"


def test_jvp_zero_output(mocker):
    g = LogicalGraph()
    n1 = LogicalNode("n1", "Input", shape_metadata=(2,))
    n2 = LogicalNode("n2", "Mul", inputs=["n1", "n1"])
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp", return_value=lambda graph, node, tangents: "out_t")
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff._get_input_tangents", return_value=["t1", "t1"])
    g.nodes["t1"] = LogicalNode("t1", "Input", [])
    g2 = jvp(g, ["n1"], ["t1"], ["n2", "n1"])
    assert len(g2.outputs) == 2
