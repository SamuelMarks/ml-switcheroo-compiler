"""Graph export utilities."""

import typing

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def _format_node(curr_id: str, node: object) -> str:  # pragma: no cover
    """Function docstring."""
    op_type = getattr(node, "op_type", "Unknown")
    label = f"{op_type}\\n{curr_id[:8]}"
    return f'  "{curr_id}" [label="{label}"];'


def _format_edges(curr_id: str, node: object, graph: object, visited: set, queue: list) -> list:  # pragma: no cover
    """Function docstring."""
    lines = []
    for inp in getattr(node, "inputs", []):
        inp_id = getattr(inp, "id", str(inp))
        lines.append(f'  "{inp_id}" -> "{curr_id}";')
        if inp_id in graph.nodes and inp_id not in visited:
            queue.append(inp_id)
            visited.add(inp_id)
    return lines


def export_to_dot(file: typing.Union[str, typing.IO], *arrays: Tensor, **kwargs: object) -> None:  # pragma: no cover
    """Exports the computation graph of the given arrays to a DOT format file.

    Args:
        file (Union[str, IO]): The file path or file-like object to write to.
        *arrays (Tensor): The output tensors to trace back from.
        **kwargs: Additional keyword arguments.
    """
    graph = global_tracing_state.active_graph
    if graph is None:
        raise RuntimeError("No active graph to export. Must be in tracing mode.")

    lines = ["digraph G {"]
    visited = set()
    queue = []

    for arr in arrays:
        node_id = getattr(getattr(arr, "data", None), "id", None)
        if node_id and node_id in graph.nodes and node_id not in visited:
            queue.append(node_id)
            visited.add(node_id)

    while queue:
        curr_id = queue.pop(0)
        node = graph.nodes[curr_id]
        lines.append(_format_node(curr_id, node))
        lines.extend(_format_edges(curr_id, node, graph, visited, queue))

    lines.append("}")
    dot_content = "\n".join(lines)

    if isinstance(file, str):
        with open(file, "w") as f:
            f.write(dot_content)
    else:
        file.write(dot_content)


__all__ = ["export_to_dot"]
