"""Test module."""

import pytest

from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort


class DummyNode:
    def __init__(self, id, inputs=None):
        self.id = id
        self.inputs = inputs or []


class DummyGraph:
    def __init__(self, nodes):
        self.nodes = {n.id: n for n in nodes}


def test_graph_utils():
    n1 = DummyNode("1")
    n2 = DummyNode("2", inputs=["1"])
    n3 = DummyNode("3", inputs=["2", "missing"])
    g = DummyGraph([n1, n2, n3])

    sorted_nodes = topological_sort(g)
    assert len(sorted_nodes) == 3
    assert sorted_nodes[0].id == "1"
    assert sorted_nodes[1].id == "2"
    assert sorted_nodes[2].id == "3"

    n_cycle = DummyNode("c", inputs=["c"])
    g_cycle = DummyGraph([n_cycle])
    with pytest.raises(CompilationError, match="Cycle detected"):
        topological_sort(g_cycle)


def test_graph_utils_already_visited():
    n1 = DummyNode("1")
    n2 = DummyNode("2", inputs=["1"])
    n3 = DummyNode("3", inputs=["1"])
    g = DummyGraph([n1, n2, n3])
    # The nodes loop will see n1 already visited when it gets to n3, hitting the if branch
    topological_sort(g)


def test_graph_utils_empty():
    g = DummyGraph([])
    assert len(topological_sort(g)) == 0


def test_graph_utils_multiple_visits():
    n1 = DummyNode("1")
    n2 = DummyNode("2", inputs=["1"])

    # If DummyGraph orders nodes such that "2" comes first, then "1" gets visited via "2".
    # Then when the outer loop reaches "1", it is already visited.
    class DeterministicGraph:
        nodes = {"2": n2, "1": n1}

    g = DeterministicGraph()
    assert len(topological_sort(g)) == 2
