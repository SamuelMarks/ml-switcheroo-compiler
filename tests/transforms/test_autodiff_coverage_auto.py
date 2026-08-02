"""Coverage tests for autodiff."""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.autodiff import _accumulate_gradients
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
