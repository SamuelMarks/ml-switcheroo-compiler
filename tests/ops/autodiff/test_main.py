"""Tests for test_main."""

from __future__ import annotations

import typing

# ruff: noqa: E501
import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode
from pytest_mock import MockerFixture

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.core.errors import MissingJVPRuleError
from ml_switcheroo_compiler.transforms.autodiff import (
    _accumulate_gradients,
    _add_nodes,
    _backward_pass,
    _copy_graph,
    _extract_gradients,
    _get_input_tangents,
    _get_reachable_from_output,
    _invoke_jvp_rule,
    _process_jvp_node,
    grad,
    hvp,
    jvp,
)
from tests.testing_utils import require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_add_nodes() -> None:
    """Test function test_add_nodes."""
    g = LogicalGraph()
    n1 = LogicalNode("n1", "Input", shape_metadata=(2,))
    n2 = LogicalNode("n2", "Input", shape_metadata=(2,))
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    out_id = _add_nodes(g, "n1", "n2")
    assert out_id in g.nodes
    assert g.nodes[out_id].op_type == "Add"


def test_copy_graph() -> None:
    """Test function test_copy_graph."""
    g = LogicalGraph(name="test")
    n1 = LogicalNode("n1", "Input", shape_metadata=(2,))
    g.nodes["n1"] = n1
    g2 = _copy_graph(g)
    assert g2.name == "test_grad"
    assert "n1" in g2.nodes


def test_get_reachable_from_output() -> None:
    """Test function test_get_reachable_from_output."""
    n1 = LogicalNode("n1", "Input", inputs=[])
    n2 = LogicalNode("n2", "Add", inputs=["n1", "n1"])
    n3 = LogicalNode("n3", "Mul", inputs=["n2", "n2"])
    assert _get_reachable_from_output([n1, n2, n3], "n3") == {"n3", "n2", "n1"}


def test_accumulate_gradients(mocker: MockerFixture) -> None:
    """Test function test_accumulate_gradients.

    Args:
        mocker: Test parameter `mocker`.
    """
    g = LogicalGraph()
    n1 = LogicalNode("n1", "Mul", inputs=["a", "b"])
    mock_get_vjp = mocker.patch("ml_switcheroo_compiler.transforms.autodiff.get_vjp")

    def dummy_vjp(graph: BackendArg, node: BackendArg, adj: BackendArg) -> BackendArg:
        """Test function dummy_vjp.

        Args:
            graph: Test parameter `graph`.
            node: Test parameter `node`.
            adj: Test parameter `adj`.

        Returns:
            BackendArg: Computed output value from `dummy_vjp`.
        """
        return ["da", "db"]

    pytest.MonkeyPatch().setattr(mock_get_vjp, "return_value", lambda g, n, adj: ["da", "db"], raising=False)
    adj: dict[str, str] = {}
    _accumulate_gradients(g, n1, "adj", adj)
    assert adj == {"a": "da", "b": "db"}
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff._add_nodes", return_value="added")
    _accumulate_gradients(g, n1, "adj", adj)
    assert adj == {"a": "added", "b": "added"}
    pytest.MonkeyPatch().setattr(mock_get_vjp, "side_effect", ValueError, raising=False)
    from ml_switcheroo_compiler.core.errors import MissingJVPRuleError

    with pytest.raises(MissingJVPRuleError):
        _accumulate_gradients(g, n1, "adj", adj)
    pytest.MonkeyPatch().setattr(mock_get_vjp, "side_effect", None, raising=False)
    pytest.MonkeyPatch().setattr(mock_get_vjp, "return_value", lambda g, n, adj: ["da"], raising=False)
    with pytest.raises(ValueError):
        _accumulate_gradients(g, n1, "adj", adj)


def test_backward_pass(mocker: MockerFixture) -> None:
    """Test function test_backward_pass.

    Args:
        mocker: Test parameter `mocker`.
    """
    g = LogicalGraph()
    n1 = LogicalNode("n1", "Mul", inputs=["a", "b"])
    n2 = LogicalNode("n2", "Input", inputs=[])
    mock_acc = mocker.patch("ml_switcheroo_compiler.transforms.autodiff._accumulate_gradients")
    _backward_pass(g, [n1, n2], {"n1"}, {"n1": "adj"})
    mock_acc.assert_called_once()
    mock_acc.reset_mock()
    _backward_pass(g, [n1], set(), {"n1": "adj"})
    mock_acc.assert_not_called()
    mock_acc.reset_mock()
    _backward_pass(g, [n2], {"n2"}, {"n2": "adj"})
    mock_acc.assert_not_called()


def test_extract_gradients() -> None:
    """Test function test_extract_gradients."""
    g = LogicalGraph()
    n1 = LogicalNode("n1", "Input", shape_metadata=(2,))
    g.nodes["n1"] = n1
    assert _extract_gradients(g, ["n1"], {"n1": "adj"}) == ["adj"]
    res = _extract_gradients(g, ["n1"], {})
    assert len(res) == 1
    assert "grad_zeros" in res[0]
    with pytest.raises(ValueError):
        _extract_gradients(g, ["not_exist"], {})


def test_grad(mocker: MockerFixture) -> None:
    """Test function test_grad.

    Args:
        mocker: Test parameter `mocker`.
    """
    g = LogicalGraph()
    n1 = LogicalNode("n1", "Input", shape_metadata=(2,))
    n2 = LogicalNode("n2", "Mul", inputs=["n1", "n1"])
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    mock_get_vjp = mocker.patch("ml_switcheroo_compiler.transforms.autodiff.get_vjp")

    def dummy_vjp(graph: BackendArg, node: BackendArg, adj: BackendArg) -> BackendArg:
        """Test function dummy_vjp.

        Args:
            graph: Test parameter `graph`.
            node: Test parameter `node`.
            adj: Test parameter `adj`.

        Returns:
            BackendArg: Computed output value from `dummy_vjp`.
        """
        return ["da", "db"]

    pytest.MonkeyPatch().setattr(mock_get_vjp, "return_value", lambda g, n, adj: ["da", "db"], raising=False)
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff._add_nodes", return_value="added")
    with pytest.raises(ValueError):
        grad(g, ["n1"], "not_exist")
    g2 = grad(g, ["n1"], "n2")
    assert g2 is not None


def test_get_input_tangents() -> None:
    """Test function test_get_input_tangents."""
    g = LogicalGraph()
    n1 = LogicalNode("n1", "Input", shape_metadata=(2,))
    g.nodes["n1"] = n1
    n2 = LogicalNode("n2", "Mul", inputs=["n1", "n3"])
    g.nodes["n3"] = LogicalNode("n3", "Input", shape_metadata=())
    res = _get_input_tangents(g, n2, {"n1": "t1"})
    assert res[0] == "t1"
    assert "jvp_zeros" in res[1]


def test_invoke_jvp_rule() -> None:
    """Test function test_invoke_jvp_rule."""
    dummy_graph = LogicalGraph()
    dummy_node = LogicalNode("n1", "Add")

    def mock_jvp(
        graph: LogicalGraph,
        node: LogicalNode,
        tangent: typing.Sequence[str] | str,
    ) -> str:
        """Test function mock_jvp.

        Args:
            graph: Test parameter `graph`.
            node: Test parameter `node`.
            tangent: Test parameter `tangent`.

        Returns:
            str: Computed output value from `mock_jvp`.
        """
        del graph, node, tangent
        return "res"

    assert _invoke_jvp_rule(mock_jvp, dummy_graph, dummy_node, ["t"]) == "res"

    def mock_jvp_bad() -> None:
        """Test function mock_jvp_bad."""
        pass

    import pytest

    with pytest.raises(MissingJVPRuleError):
        require_callable(_invoke_jvp_rule)(mock_jvp_bad, None, None, ["t"])


def test_process_jvp_node(mocker: MockerFixture) -> None:
    """Test function test_process_jvp_node.

    Args:
        mocker: Test parameter `mocker`.
    """
    g = LogicalGraph()
    n1 = LogicalNode("n1", "Mul", inputs=["a", "b"])
    mock_get_jvp = mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp")
    pytest.MonkeyPatch().setattr(mock_get_jvp, "return_value", lambda graph, node, tangents: "out_t", raising=False)
    _process_jvp_node(g, n1, {})
    tangents = {"a": "t1"}
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff._get_input_tangents", return_value=["t1", "t2"])
    _process_jvp_node(g, n1, tangents)
    assert tangents["n1"] == "out_t"
    pytest.MonkeyPatch().setattr(mock_get_jvp, "side_effect", ValueError, raising=False)
    with pytest.raises(ValueError):
        _process_jvp_node(g, n1, tangents)


def test_jvp_hvp(mocker: MockerFixture) -> None:
    """Test function test_jvp_hvp.

    Args:
        mocker: Test parameter `mocker`.
    """
    g = LogicalGraph()
    n1 = LogicalNode("n1", "Input", shape_metadata=(2,))
    n2 = LogicalNode("n2", "Mul", inputs=["n1", "n1"])
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    mock_get_jvp = mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp")
    pytest.MonkeyPatch().setattr(mock_get_jvp, "return_value", lambda graph, node, tangents: "out_t", raising=False)
    with pytest.raises(ValueError):
        jvp(g, ["n1"], ["t1", "t2"], ["n2"])
    with pytest.raises(ValueError):
        jvp(g, ["n1"], ["t1"], ["not_exist"])
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff._get_input_tangents", return_value=["t1", "t1"])
    g2 = jvp(g, ["n1"], ["t1"], ["n2"])
    assert g2 is not None
    hvp_graph = LogicalGraph(name="hvp_res")
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff.grad", return_value=g2)
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry.has_vjp", return_value=True)
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.has_jvp", return_value=True)
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff.jvp", return_value=hvp_graph)
    assert hvp(g, ["n1"], ["t1"], ["n2"]) is hvp_graph


def test_jvp_zero_output(mocker: MockerFixture) -> None:
    """Test function test_jvp_zero_output.

    Args:
        mocker: Test parameter `mocker`.
    """
    g = LogicalGraph()
    n1 = LogicalNode("n1", "Input", shape_metadata=(2,))
    n2 = LogicalNode("n2", "Mul", inputs=["n1", "n1"])
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    mocker.patch(
        "ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp",
        return_value=lambda graph, node, tangents: "out_t",
    )
    mocker.patch("ml_switcheroo_compiler.transforms.autodiff._get_input_tangents", return_value=["t1", "t1"])
    g.nodes["t1"] = LogicalNode("t1", "Input", inputs=[])
    g2 = jvp(g, ["n1"], ["t1"], ["n2", "n1"])
    assert len(g2.outputs) == 2
