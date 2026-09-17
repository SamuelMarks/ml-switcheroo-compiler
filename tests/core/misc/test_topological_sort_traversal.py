"""Unit tests for graph topological sorting traversal and cycle detection."""

from __future__ import annotations

import pytest

from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort


class _GraphNode:
    """Mock graph node representation for topological sort tests."""

    def __init__(self, id: str, inputs: list[str] | None = None) -> None:
        """Initialize graph node.

        Args:
            id (str): Unique node identifier.
            inputs (list[str] | None): List of upstream input node identifiers.
        """
        self.id: str = id
        self.inputs: list[str] = inputs or []


class _GraphContainer:
    """Mock graph container storing nodes in a dictionary."""

    def __init__(self, nodes: list[_GraphNode]) -> None:
        """Initialize graph container.

        Args:
            nodes (list[_GraphNode]): List of graph nodes.
        """
        self.nodes: dict[str, _GraphNode] = {n.id: n for n in nodes}


def test_topological_sort_dependencies() -> None:
    """Verify topological sort respects dependency ordering and detects cycles."""
    n1 = _GraphNode("1")
    n2 = _GraphNode("2", inputs=["1"])
    n3 = _GraphNode("3", inputs=["2", "missing"])
    g = _GraphContainer([n1, n2, n3])

    sorted_nodes = topological_sort(g)
    assert len(sorted_nodes) == 3
    assert sorted_nodes[0].id == "1"
    assert sorted_nodes[1].id == "2"
    assert sorted_nodes[2].id == "3"

    n_cycle = _GraphNode("c", inputs=["c"])
    g_cycle = _GraphContainer([n_cycle])
    with pytest.raises(CompilationError, match="Cycle detected"):
        topological_sort(g_cycle)


def test_topological_sort_visited_nodes() -> None:
    """Verify nodes shared across multiple dependent paths are visited once."""
    n1 = _GraphNode("1")
    n2 = _GraphNode("2", inputs=["1"])
    n3 = _GraphNode("3", inputs=["1"])
    g = _GraphContainer([n1, n2, n3])
    sorted_nodes = topological_sort(g)
    assert len(sorted_nodes) == 3


def test_topological_sort_empty_graph() -> None:
    """Verify topological sort returns empty list for an empty graph container."""
    g = _GraphContainer([])
    assert len(topological_sort(g)) == 0


def test_topological_sort_multiple_visits() -> None:
    """Verify deterministic resolution when root nodes appear after dependents."""
    n1 = _GraphNode("1")
    n2 = _GraphNode("2", inputs=["1"])

    class DeterministicGraph:
        """Deterministic graph with specific key iteration order."""

        nodes: dict[str, _GraphNode] = {"2": n2, "1": n1}

    g = DeterministicGraph()
    assert len(topological_sort(g)) == 2
