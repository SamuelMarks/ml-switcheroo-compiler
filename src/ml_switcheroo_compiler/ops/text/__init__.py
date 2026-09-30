# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Text operations module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ml_switcheroo_compiler.ops.text import ops
from ml_switcheroo_compiler.ops.text.frontend import (
    AsStringConfig,
    as_string,
    edit_distance,
    lookup,
    regex_full_match,
    regex_replace,
    string_join,
    string_length,
    string_lower,
    string_split,
    string_substr,
    string_to_hash,
    string_to_number,
    string_upper,
    text_vectorization,
)

if TYPE_CHECKING:
    from ml_switcheroo_compiler.ir.core import IRNode

_ = ops

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("CreateToken")
class CreateToken(OpDef):
    """CreateToken operation."""

    op_name = "CreateToken"


def create_token(
    *args: str | int | float | bool,
    **kwargs: str | int | float | bool | None,
) -> IRNode:
    """Create token.

    Args:
        *args (str | int | float | bool): Positional args.
        **kwargs (str | int | float | bool | None): Keyword args.

    Returns:
        IRNode: Resulting graph node.
    """
    from ml_switcheroo_compiler.ops.base import get_op

    return get_op("CreateToken")()(*args, **kwargs)


__all__ = [
    "AsStringConfig",
    "CreateToken",
    "as_string",
    "create_token",
    "edit_distance",
    "lookup",
    "regex_full_match",
    "regex_replace",
    "string_join",
    "string_length",
    "string_lower",
    "string_split",
    "string_substr",
    "string_to_hash",
    "string_to_number",
    "string_upper",
    "text_vectorization",
]
