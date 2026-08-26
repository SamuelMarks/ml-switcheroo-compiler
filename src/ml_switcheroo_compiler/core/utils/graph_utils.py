# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Graph utilities."""

from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.ir.core import IRGraph


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
        self.sorted_nodes = []

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

    def sort(self):
        """Perform the topological sort.

        Returns:
            list[object]: The sorted nodes.
        """
        nodes_iterable = self.graph.nodes
        if isinstance(nodes_iterable, dict):
            nodes_iterable = list(nodes_iterable.keys())
        for n in nodes_iterable:
            node_id = getattr(n, "id", n) if not isinstance(n, str) else n
            if node_id not in self.visited:
                self.visit(node_id)
        return self.sorted_nodes


def topological_sort(graph: IRGraph):
    """Perform topological sort on a graph.

    Args:
        graph: LogicalGraph or IRGraph

    Returns:
        List of sorted nodes
    """
    return _TopologicalSorter(graph).sort()
