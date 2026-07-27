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

__all__ = [
    "AsStringConfig",
    "as_string",
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

_ = ops
