import pytest

from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort


class MockNode:
    def __init__(self, id, inputs):
        self.id = id
        self.inputs = inputs


class MockGraphDict:
    def __init__(self, nodes):
        self.nodes = nodes


class MockGraphList:
    def __init__(self, nodes):
        self.nodes = nodes


def test_topological_sort_dict():
    n1 = MockNode("n1", [])
    n2 = MockNode("n2", ["n1"])
    n3 = MockNode("n3", ["n2"])

    graph = MockGraphDict({"n3": n3, "n1": n1, "n2": n2})
    sorted_nodes = topological_sort(graph)
    assert [n.id for n in sorted_nodes] == ["n1", "n2", "n3"]


def test_topological_sort_list():
    n1 = MockNode("n1", [])
    n2 = MockNode("n2", ["n1"])
    n3 = MockNode("n3", ["n2"])

    graph = MockGraphList([n3, n1, n2])
    sorted_nodes = topological_sort(graph)
    assert [n.id for n in sorted_nodes] == ["n1", "n2", "n3"]


def test_topological_sort_cycle():
    n1 = MockNode("n1", ["n3"])
    n2 = MockNode("n2", ["n1"])
    n3 = MockNode("n3", ["n2"])

    graph = MockGraphDict({"n1": n1, "n2": n2, "n3": n3})
    with pytest.raises(CompilationError, match="Cycle detected in graph."):
        topological_sort(graph)


def test_topological_sort_missing_node():
    n1 = MockNode("n1", ["n_missing"])
    graph = MockGraphDict({"n1": n1})
    sorted_nodes = topological_sort(graph)
    # The missing node won't crash, it just visits and adds to visited and ignores adding to sorted list.
    assert [n.id for n in sorted_nodes] == ["n1"]


def test_topological_sort_already_visited():
    n1 = MockNode("n1", [])
    n2 = MockNode("n2", ["n1"])
    n3 = MockNode("n3", ["n1"])
    n4 = MockNode("n4", ["n2", "n3"])
    graph = MockGraphDict({"n1": n1, "n2": n2, "n3": n3, "n4": n4})
    sorted_nodes = topological_sort(graph)
    assert len(sorted_nodes) == 4
