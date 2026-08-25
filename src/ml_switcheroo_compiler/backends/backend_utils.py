"""Module backend_utils.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Backend utilities."""


from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


def format_shape_metadata(node: IRNode, var_names: dict[str, str]) -> str | None:
    """Format shape metadata for a node.

    Args:
        node (IRNode): The node to process.
        var_names (dict[str, str]): Variable name mapping.

    Returns:
        str | None: Formatted shape metadata.
    """
    if not (hasattr(node, "shape_metadata") and node.shape_metadata):
        return None
    formatted_shape: object = []
    for dim in node.shape_metadata:
        if hasattr(dim, "id"):
            formatted_shape.append(var_names.get(dim.id, dim.id))
        elif isinstance(dim, str):
            formatted_shape.append(f"'{dim}'")
        else:
            formatted_shape.append(str(dim))
    return f"({', '.join(formatted_shape)}{',' if len(formatted_shape) == 1 else ''})"
