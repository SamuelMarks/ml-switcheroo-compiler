"""Backend utilities."""

from typing import Optional

from ml_switcheroo_compiler.ir.core import IRNode


def resolve_input_vars(node: IRNode, var_names: dict[str, str]) -> list[str]:
    """Resolve input variable names for a node.

    Args:
        node (IRNode): The node to process.
        var_names (dict[str, str]): Variable name mapping.

    Returns:
        list[str]: Resolved variable names.
    """
    return [var_names.get(in_id, in_id) for in_id in node.inputs]


def format_shape_metadata(node: IRNode, var_names: dict[str, str]) -> Optional[str]:
    """Format shape metadata for a node.

    Args:
        node (IRNode): The node to process.
        var_names (dict[str, str]): Variable name mapping.

    Returns:
        Optional[str]: Formatted shape metadata.
    """
    if not (hasattr(node, "shape_metadata") and node.shape_metadata):
        return None
    formatted_shape = []
    for dim in node.shape_metadata:
        if hasattr(dim, "id"):  # pragma: no branch
            formatted_shape.append(var_names.get(dim.id, dim.id))  # pragma: no cover
        elif isinstance(dim, str):  # pragma: no branch
            formatted_shape.append(f"'{dim}'")  # pragma: no cover
        else:
            formatted_shape.append(str(dim))
    return f"({', '.join(formatted_shape)}{',' if len(formatted_shape) == 1 else ''})"
