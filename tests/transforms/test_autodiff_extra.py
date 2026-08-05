from unittest import mock

from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode
from ml_switcheroo_compiler.transforms.autodiff import jvp


def test_jvp_returns_none():
    graph = IRGraph()
    graph.outputs = ["dummy_id"]
    graph.nodes = {"inp": LogicalNode("inp", "Input"), "dummy_id": LogicalNode("dummy_id", "DummyJVPNode", inputs=["inp"])}

    with mock.patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp") as mock_get:
        mock_get.return_value = lambda graph, node, tangents: None
        jvp(graph, ["inp"], ["tangent_inp"], ["dummy_id"])
