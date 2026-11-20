"""Public package exports and initialization for `ml_switcheroo_compiler.ops.text`."""

from __future__ import annotations

from ml_switcheroo_compiler.core.shape import OpValue, extract_shape
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.text import ops
from ml_switcheroo_compiler.ops.text.api import (
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


@register_op("CreateToken")
class CreateToken(OpDef):
    """CreateToken operation."""

    op_name = "CreateToken"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple: Shape tuple.
        """
        a = args[0]
        shape = extract_shape(a)
        max_length = kwargs.get("max_length")
        if max_length is not None:
            return shape + (max_length,)
        return shape


def create_token(
    *args: str | int | float | bool,
    **kwargs: str | int | float | bool | None,
) -> OpValue:
    """Create token.

    Args:
        *args (str | int | float | bool): Positional args.
        **kwargs (str | int | float | bool | None): Keyword args.

    Returns:
        OpValue: Resulting token tensor or graph node.
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
