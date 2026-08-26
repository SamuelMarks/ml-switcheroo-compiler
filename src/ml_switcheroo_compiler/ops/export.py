# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Graph export utilities."""

import typing

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.tracing.state import global_tracing_state


class _DotGraphVisitor:
    """Visitor for DOT graph export."""

    def __init__(self, graph) -> None:
        """Initialize the visitor.

        Args:
            graph (object): The IR graph.
        """
        self.graph = graph
        self.visited = set()
        self.lines = ["digraph G {"]

    def visit(self, node_id: str) -> None:
        """Visit a node.

        Args:
        node_id (str): The node_id parameter.

        Returns:
        NoneType: Result.
        """
        if not node_id or node_id not in self.graph.nodes or node_id in self.visited:
            return

        self.visited.add(node_id)
        node = self.graph.nodes[node_id]

        # Format the node
        op_type = getattr(node, "op_type", "Unknown")
        label = f"{op_type}\\n{node_id[:8]}"
        self.lines.append(f'  "{node_id}" [label="{label}"];')

        # Format the edges and recurse
        for inp in getattr(node, "inputs", []):
            inp_id = getattr(inp, "id", str(inp))
            self.lines.append(f'  "{inp_id}" -> "{node_id}";')
            self.visit(inp_id)


def export_to_dot(file: typing.Union[str, typing.IO], *arrays: Tensor, **kwargs) -> None:
    """Export the computation graph of the given arrays to a DOT format file.

    Args:
        file (object): The file parameter.
        *arrays (Tensor): Positional args.
        **kwargs (object): Keyword args.

    Raises:
        RuntimeError: An exception.
    """
    graph = global_tracing_state.active_graph
    if graph is None:
        raise RuntimeError("No active graph to export. Must be in tracing mode.")

    visitor = _DotGraphVisitor(graph)

    for arr in arrays:
        node_id = getattr(getattr(arr, "data", None), "id", None)
        visitor.visit(node_id)

    visitor.lines.append("}")
    dot_content = "\n".join(visitor.lines)

    if isinstance(file, str):
        with open(file, "w") as f:
            f.write(dot_content)
    else:
        file.write(dot_content)


__all__ = ["export_to_dot"]
