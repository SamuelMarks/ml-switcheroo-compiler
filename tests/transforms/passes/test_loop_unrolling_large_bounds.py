"""Tests for test_loop_unrolling_large_bounds."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_loop_unrolling_missing_coverage() -> None:
    """Test function test_loop_unrolling_missing_coverage."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode
    from ml_switcheroo_compiler.transforms.passes.loop_unrolling import _load_config, detect_static_bound, unroll_loops

    with patch("os.path.exists", return_value=False):
        assert _load_config() == {}

    class MockNode:
        """Test class MockNode."""

        op_type = "Add"
        id = "mock"

    assert cast("EagerCallableProtocol", detect_static_bound)(cast("EagerCallableProtocol", MockNode)(), []) is None

    g = IRGraph()
    n1 = LogicalNode(id="n1", op_type="Add")
    g.nodes["n1"] = n1
    unrolled_g = unroll_loops(g)
    assert "n1" in unrolled_g.nodes


def test_clone_subgraph_coverage() -> None:
    """Test function test_clone_subgraph_coverage."""
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode
    from ml_switcheroo_compiler.transforms.passes.loop_unrolling import clone_subgraph

    g = IRGraph()
    n1 = LogicalNode(id="n1", op_type="Add", inputs=["in1"])
    g.nodes["n1"] = n1

    cloned = clone_subgraph(g, "prefix", {"in1": "new_in1"})
    assert len(cloned) == 1
    assert cloned[0].id == "prefix_n1"
    assert cloned[0].inputs == ["new_in1"]
