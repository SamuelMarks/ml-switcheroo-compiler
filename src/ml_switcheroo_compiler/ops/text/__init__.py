# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Text operations module."""

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

_ = ops

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("CreateToken")
class CreateToken(OpDef):
    """CreateToken operation."""

    op_name = "CreateToken"


def create_token(*args: Any, **kwargs: Any) -> Any:
    """Create token.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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
