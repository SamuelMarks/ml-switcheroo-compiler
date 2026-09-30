"""Backend code generation and node formatting utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml_switcheroo_compiler.ir.core import IRNode


def resolve_input_vars(node: IRNode, var_names: dict[str, str]) -> list[str]:
    """Resolve input variable names for an IR node.

    Args:
        node (IRNode): The IR node whose inputs are resolved.
        var_names (dict[str, str]): Variable name mapping dictionary.

    Returns:
        list[str]: Resolved variable names in the target emitter context.
    """
    return [var_names.get(in_id, in_id) for in_id in node.inputs]


def format_shape_metadata(node: IRNode, var_names: dict[str, str]) -> str | None:
    """Format shape metadata for an IR node into a code representation.

    Args:
        node (IRNode): The IR node containing shape metadata.
        var_names (dict[str, str]): Variable name mapping dictionary.

    Returns:
        str | None: Formatted shape tuple representation or None if not present.
    """
    if not (hasattr(node, "shape_metadata") and node.shape_metadata):
        return None
    formatted_shape: list[str] = []
    for dim in node.shape_metadata:
        if hasattr(dim, "id"):
            dim_id = str(dim.id)
            formatted_shape.append(var_names.get(dim_id, dim_id))
        elif isinstance(dim, str):
            formatted_shape.append(f"'{dim}'")
        else:
            formatted_shape.append(str(dim))
    return f"({', '.join(formatted_shape)}{',' if len(formatted_shape) == 1 else ''})"
