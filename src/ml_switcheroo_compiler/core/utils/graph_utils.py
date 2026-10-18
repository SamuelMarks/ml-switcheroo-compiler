"""Graph utilities."""

from __future__ import annotations

from collections.abc import Iterable

from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode


class _TopologicalSorter:
    """Help class for topological sorting."""

    def __init__(self, graph: IRGraph) -> None:
        """Initialize the sorter.

        Args:
            graph: The graph to sort.
        """
        self.graph = graph
        self.visited: set[str] = set()
        self.temp_mark: set[str] = set()
        self.sorted_nodes: list[LogicalNode] = []

    def visit(self, node_id: str) -> None:
        """Visit a node during sorting.

        Args:
            node_id (str): The node ID to visit.

        Raises:
            CompilationError: If a cycle is detected.
        """
        if node_id in self.temp_mark:
            msg = "Cycle detected in graph."
            raise CompilationError(msg)

        if node_id in self.visited:
            return

        self.temp_mark.add(node_id)
        node = None
        if isinstance(self.graph.nodes, dict):
            node = self.graph.nodes.get(node_id)
        elif isinstance(self.graph.nodes, list):
            for n in self.graph.nodes:
                if getattr(n, "id", "") == node_id:
                    node = n
                    break

        if node is not None:
            for in_id in node.inputs:
                self.visit(in_id)
            self.temp_mark.remove(node_id)
            self.visited.add(node_id)
            self.sorted_nodes.append(node)
        else:
            self.temp_mark.remove(node_id)
            self.visited.add(node_id)

    def sort(self) -> list[LogicalNode]:
        """Perform the topological sort.

        Returns:
            list[object]: The sorted nodes.
        """
        nodes_iterable: Iterable[str | LogicalNode] = self.graph.nodes
        if isinstance(nodes_iterable, dict):
            nodes_iterable = list(nodes_iterable.keys())
        for n in nodes_iterable:
            node_id = getattr(n, "id", n) if not isinstance(n, str) else n
            if node_id not in self.visited:
                self.visit(str(node_id))
        return self.sorted_nodes


def topological_sort(graph: IRGraph) -> list[LogicalNode]:
    """Perform topological sort on a graph.

    Args:
        graph: LogicalGraph or IRGraph

    Returns:
        List of sorted nodes
    """
    return _TopologicalSorter(graph).sort()
