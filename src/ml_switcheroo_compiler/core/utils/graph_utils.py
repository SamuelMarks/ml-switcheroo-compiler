# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Graph utilities."""

from typing import Any

from ml_switcheroo_compiler.core.errors import CompilationError


class _TopologicalSorter:
    """Help class for topological sorting."""

    def __init__(self, graph: Any) -> None:
        """Initialize the sorter.

        Args:
            graph: The graph to sort.
        """
        self.graph = graph
        self.visited: set[str] = set()
        self.temp_mark: set[str] = set()
        self.sorted_nodes: list[Any] = []

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
        node = self.graph.nodes.get(node_id)

        if node is not None:
            for in_id in node.inputs:
                self.visit(in_id)
            self.temp_mark.remove(node_id)
            self.visited.add(node_id)
            self.sorted_nodes.append(node)
        else:
            self.temp_mark.remove(node_id)
            self.visited.add(node_id)

    def sort(self) -> list[Any]:
        """Perform the topological sort.

        Returns:
            list[Any]: The sorted nodes.
        """
        for node_id in self.graph.nodes:
            if node_id not in self.visited:
                self.visit(node_id)
        return self.sorted_nodes


def topological_sort(graph: Any) -> list[Any]:
    """Perform topological sort on a graph.

    Args:
        graph: LogicalGraph or IRGraph

    Returns:
        List of sorted nodes
    """
    return _TopologicalSorter(graph).sort()
