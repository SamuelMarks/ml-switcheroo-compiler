"""Graph utilities."""

from __future__ import annotations

import typing
from collections.abc import Mapping

from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.core.shape import TensorOrScalar, is_traced_data
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode, as_node_list


@typing.runtime_checkable
class _NodeGetProtocol(typing.Protocol):
    """Protocol for node containers supporting `.get(node_id)`."""

    def get(self, key: str) -> LogicalNode | None:
        """Retrieve node by key.

        Args:
            key (str): Node key.

        Returns:
            LogicalNode | None: Retrieved node or None.
        """
        ...


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
        node: LogicalNode | None = None
        raw_nodes: TensorOrScalar = self.graph.nodes
        if isinstance(raw_nodes, _NodeGetProtocol):
            node = raw_nodes.get(node_id)
        else:
            for n in as_node_list(raw_nodes):
                if n.id == node_id:
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
            list[LogicalNode]: The sorted nodes.
        """
        raw_nodes: TensorOrScalar = self.graph.nodes
        node_ids: list[str]
        if isinstance(raw_nodes, Mapping):
            node_ids = [str(k) for k in raw_nodes.keys()]
        elif isinstance(raw_nodes, list):
            node_ids = [x if isinstance(x, str) else (str(x.id) if isinstance(x, LogicalNode) or is_traced_data(x) else str(x)) for x in raw_nodes]
        else:
            return []
        for node_id in node_ids:
            if node_id not in self.visited:
                self.visit(node_id)
        return self.sorted_nodes


def topological_sort(graph: IRGraph) -> list[LogicalNode]:
    """Perform topological sort on a graph.

    Args:
        graph: LogicalGraph or IRGraph

    Returns:
        List of sorted nodes
    """
    return _TopologicalSorter(graph).sort()
