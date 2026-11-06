"""Backend code generation and node formatting utilities."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.shape import (
    ShapeMetadataProtocol,
    TensorOrScalar,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.ir.core import IRNode


@typing.runtime_checkable
class _DimWithIdProtocol(typing.Protocol):
    """Protocol for symbolic dimension objects exposing `.id`."""

    id: str | TensorOrScalar


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
    raw_node: TensorOrScalar = node
    if not (isinstance(raw_node, ShapeMetadataProtocol) and raw_node.shape_metadata):
        return None
    formatted_shape: list[str] = []
    for dim in raw_node.shape_metadata:
        raw_dim: TensorOrScalar = dim
        if isinstance(raw_dim, _DimWithIdProtocol):
            dim_id = str(raw_dim.id)
            formatted_shape.append(var_names.get(dim_id, dim_id))
        elif isinstance(dim, str):
            formatted_shape.append(f"'{dim}'")
        else:
            formatted_shape.append(str(dim))
    return f"({', '.join(formatted_shape)}{',' if len(formatted_shape) == 1 else ''})"
