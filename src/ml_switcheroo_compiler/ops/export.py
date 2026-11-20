"""Graph export utilities."""

from __future__ import annotations

import typing
from collections.abc import Mapping

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.shape import OpValue, extract_data, is_traced_data
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.tracing.state import global_tracing_state


@typing.runtime_checkable
class _DotGraphProtocol(typing.Protocol):
    """Structural protocol for graph objects exposing a ``nodes`` mapping."""

    nodes: Mapping[str, LogicalNode]


class _DotGraphVisitor:
    """Visitor for DOT graph export."""

    def __init__(self, graph: LogicalGraph | _DotGraphProtocol | OpValue) -> None:
        """Initialize the visitor.

        Args:
            graph (LogicalGraph | _DotGraphProtocol | OpValue): The IR graph to traverse.
        """
        self.nodes: Mapping[str, LogicalNode] = graph.nodes if isinstance(graph, _DotGraphProtocol) else {}
        self.visited: set[str] = set()
        self.lines = ["digraph G {"]

    def visit(self, node_id: str | None) -> None:
        """Visit a node.

        Args:
        node_id (str | None): Input `node_id` argument (node id).

        Returns:
        NoneType: Computed `NoneType` output.
        """
        if not node_id or node_id not in self.nodes or node_id in self.visited:
            return

        self.visited.add(node_id)
        node = self.nodes[node_id]

        # Format the node
        op_type = node.op_type if node.op_type else "Unknown"
        label = f"{op_type}\\n{node_id[:8]}"
        self.lines.append(f'  "{node_id}" [label="{label}"];')

        # Format the edges and recurse
        for inp in node.inputs:
            raw_inp: OpValue = inp
            inp_id = str(raw_inp.id) if is_traced_data(raw_inp) else str(inp)
            self.lines.append(f'  "{inp_id}" -> "{node_id}";')
            self.visit(inp_id)


def export_to_dot(file: str | typing.IO[str], *arrays: Tensor, **kwargs: OpValue) -> None:
    """Export the computation graph of the given arrays to a DOT format file.

    Args:
        file (str | typing.IO[str]): Input `file` argument (file).
        *arrays (Tensor): Positional args.
        **kwargs (OpValue): Keyword args.

    Raises:
        RuntimeError: If invalid inputs or unsupported runtime conditions trigger `RuntimeError`.
    """
    graph = global_tracing_state.active_graph
    if graph is None:
        raise RuntimeError("No active graph to export. Must be in tracing mode.")

    visitor = _DotGraphVisitor(graph)

    for arr in arrays:
        arr_data: OpValue = extract_data(arr)
        node_id = str(arr_data.id) if is_traced_data(arr_data) else None
        visitor.visit(node_id)

    visitor.lines.append("}")
    dot_content = "\n".join(visitor.lines)

    if isinstance(file, str):
        with open(file, "w") as f:
            f.write(dot_content)
    else:
        file.write(dot_content)


__all__ = ["export_to_dot"]
