"""Tests for test_graph_utils."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


class MockNode:
    """Test class MockNode."""

    def __init__(self, id: EagerValue, inputs: EagerValue) -> None:
        """Test function __init__.

        Args:
            id: Test parameter `id`.
            inputs: Test parameter `inputs`.
        """
        self.id = id
        self.inputs = inputs


class MockGraphDict:
    """Test class MockGraphDict."""

    def __init__(self, nodes: EagerValue) -> None:
        """Test function __init__.

        Args:
            nodes: Test parameter `nodes`.
        """
        self.nodes = nodes


class MockGraphList:
    """Test class MockGraphList."""

    def __init__(self, nodes: EagerValue) -> None:
        """Test function __init__.

        Args:
            nodes: Test parameter `nodes`.
        """
        self.nodes = nodes


def test_topological_sort_dict() -> None:
    """Test function test_topological_sort_dict."""
    n1 = MockNode("n1", [])
    n2 = MockNode("n2", ["n1"])
    n3 = MockNode("n3", ["n2"])

    graph = as_callable(MockGraphDict)({"n3": n3, "n1": n1, "n2": n2})
    sorted_nodes = as_callable(topological_sort)(graph)
    assert [n.id for n in sorted_nodes] == ["n1", "n2", "n3"]


def test_topological_sort_list() -> None:
    """Test function test_topological_sort_list."""
    n1 = MockNode("n1", [])
    n2 = MockNode("n2", ["n1"])
    n3 = MockNode("n3", ["n2"])

    graph = as_callable(MockGraphList)([n3, n1, n2])
    sorted_nodes = as_callable(topological_sort)(graph)
    assert [n.id for n in sorted_nodes] == ["n1", "n2", "n3"]


def test_topological_sort_cycle() -> None:
    """Test function test_topological_sort_cycle."""
    n1 = MockNode("n1", ["n3"])
    n2 = MockNode("n2", ["n1"])
    n3 = MockNode("n3", ["n2"])

    graph = as_callable(MockGraphDict)({"n1": n1, "n2": n2, "n3": n3})
    with pytest.raises(CompilationError, match="Cycle detected in graph."):
        as_callable(topological_sort)(graph)


def test_topological_sort_missing_node() -> None:
    """Test function test_topological_sort_missing_node."""
    n1 = MockNode("n1", ["n_missing"])
    graph = as_callable(MockGraphDict)({"n1": n1})
    sorted_nodes = as_callable(topological_sort)(graph)
    # The missing node won't crash, it just visits and adds to visited and ignores adding to sorted list.
    assert [n.id for n in sorted_nodes] == ["n1"]


def test_topological_sort_already_visited() -> None:
    """Test function test_topological_sort_already_visited."""
    n1 = MockNode("n1", [])
    n2 = MockNode("n2", ["n1"])
    n3 = MockNode("n3", ["n1"])
    n4 = MockNode("n4", ["n2", "n3"])
    graph = as_callable(MockGraphDict)({"n1": n1, "n2": n2, "n3": n3, "n4": n4})
    sorted_nodes = as_callable(topological_sort)(graph)
    assert len(sorted_nodes) == 4


def test_topological_sort_missing_node_list() -> None:
    """Test function test_topological_sort_missing_node_list."""
    n1 = MockNode("n1", ["n_missing"])
    graph = as_callable(MockGraphList)([n1])
    sorted_nodes = as_callable(topological_sort)(graph)
    assert [n.id for n in sorted_nodes] == ["n1"]


def test_topological_sort_nodes_neither_dict_nor_list() -> None:
    """Test function test_topological_sort_nodes_neither_dict_nor_list."""
    n1 = MockNode("n1", [])
    graph = as_callable(MockGraphDict)((n1,))
    sorted_nodes = as_callable(topological_sort)(graph)
    assert sorted_nodes == []
