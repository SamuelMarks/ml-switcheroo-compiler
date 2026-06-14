"""Graph utilities."""

from ml_switcheroo_compiler.core.errors import CompilationError


def topological_sort(graph: object) -> list[object]:
    """Perform topological sort on a graph.

    Args:
        graph: LogicalGraph or IRGraph

    Returns:
        List of sorted nodes
    """
    visited: set[str] = set()
    temp_mark: set[str] = set()
    sorted_nodes: list[object] = []

    def visit(node_id: str) -> None:
        """Execute visit.

        Args:
            node_id (Any): Argument node_id.
        """
        if node_id in temp_mark:
            msg = "Cycle detected in graph."
            raise CompilationError(msg)
        if node_id not in visited:
            temp_mark.add(node_id)
            node = graph.nodes.get(node_id)
            if node is not None:
                for in_id in node.inputs:
                    visit(in_id)
            temp_mark.remove(node_id)
            visited.add(node_id)
            if node is not None:
                sorted_nodes.append(node)

    for node_id in graph.nodes:
        if node_id not in visited:
            visit(node_id)

    return sorted_nodes
