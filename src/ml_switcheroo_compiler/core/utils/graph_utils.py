"""Graph utilities."""

from ml_switcheroo_compiler.core.errors import CompilationError


class _TopologicalSorter:
    """Help class for topological sorting."""

    def __init__(self, graph: object) -> None:
        """Initialize the sorter.

        Args:
            graph: The graph to sort.
        """
        self.graph = graph
        self.visited: set[str] = set()
        self.temp_mark: set[str] = set()
        self.sorted_nodes: list[object] = []

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

    def sort(self) -> list[object]:
        """Perform the topological sort.

        Returns:
            list[object]: The sorted nodes.
        """
        for node_id in self.graph.nodes:
            if node_id not in self.visited:
                self.visit(node_id)
        return self.sorted_nodes


def topological_sort(graph: object) -> list[object]:
    """Perform topological sort on a graph.

    Args:
        graph: LogicalGraph or IRGraph

    Returns:
        List of sorted nodes
    """
    return _TopologicalSorter(graph).sort()
