"""Text operations module."""

from ml_switcheroo_compiler.ops.text import ops  # noqa: F401
from ml_switcheroo_compiler.ops.text.frontend import (
    lookup,
    regex_replace,
    string_lower,
    string_split,
    string_to_hash,
    string_to_number,
    string_upper,
    text_vectorization,
    edit_distance,
    as_string,
)

__all__ = [
    "regex_replace",
    "string_split",
    "lookup",
    "string_to_hash",
    "string_to_number",
    "string_lower",
    "string_upper",
    "text_vectorization",
    "edit_distance",
    "as_string",
]
