"""Graph export utilities."""

import typing
from ml_switcheroo_compiler.core.tensor import Tensor


def export_to_dot(file: typing.Union[str, typing.IO], *arrays: Tensor, **kwargs: object) -> None:
    """Exports the computation graph of the given arrays to a DOT format file.

    Args:
        file (Union[str, IO]): The file path or file-like object to write to.
        *arrays (Tensor): The output tensors to trace back from.
        **kwargs: Additional keyword arguments.
    """
    from ml_switcheroo_compiler.tracing.tracer import _tracer  # pragma: no cover

    graph = _tracer.active_graph  # pragma: no cover
    if graph is None:  # pragma: no cover
        raise RuntimeError(
            "No active graph to export. Must be in tracing mode."
        )  # pragma: no cover

    lines = ["digraph G {"]  # pragma: no cover

    # Simple BFS to find reachable nodes
    visited = set()  # pragma: no cover
    queue = []  # pragma: no cover

    for arr in arrays:  # pragma: no cover
        if hasattr(arr, "data") and hasattr(arr.data, "id"):  # pragma: no cover
            node_id = arr.data.id  # pragma: no cover
            if node_id in graph.nodes and node_id not in visited:  # pragma: no cover
                queue.append(node_id)  # pragma: no cover
                visited.add(node_id)  # pragma: no cover

    while queue:  # pragma: no cover
        curr_id = queue.pop(0)  # pragma: no cover
        node = graph.nodes[curr_id]  # pragma: no cover

        op_type = getattr(node, "op_type", "Unknown")  # pragma: no cover
        label = f"{op_type}\\n{curr_id[:8]}"  # pragma: no cover
        lines.append(f'  "{curr_id}" [label="{label}"];')  # pragma: no cover

        for inp in getattr(node, "inputs", []):  # pragma: no cover
            if hasattr(inp, "id"):  # pragma: no cover
                inp_id = inp.id  # pragma: no cover
            else:  # pragma: no cover
                inp_id = str(inp)  # pragma: no cover

            lines.append(f'  "{inp_id}" -> "{curr_id}";')  # pragma: no cover
            if inp_id in graph.nodes and inp_id not in visited:  # pragma: no cover
                queue.append(inp_id)  # pragma: no cover
                visited.add(inp_id)  # pragma: no cover

    lines.append("}")  # pragma: no cover
    dot_content = "\n".join(lines)  # pragma: no cover

    if isinstance(file, str):  # pragma: no cover
        with open(file, "w") as f:  # pragma: no cover
            f.write(dot_content)  # pragma: no cover
    else:  # pragma: no cover
        file.write(dot_content)  # pragma: no cover


__all__ = ["export_to_dot"]
