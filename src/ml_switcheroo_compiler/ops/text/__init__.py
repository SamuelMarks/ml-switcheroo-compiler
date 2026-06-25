"""Text operations module."""

from ml_switcheroo_compiler.ops.text import ops  # noqa: F401
from ml_switcheroo_compiler.ops.text.frontend import (
    lookup,
    regex_replace,
    regex_full_match,
    string_join,
    string_length,
    string_substr,
    string_lower,
    string_split,
    string_to_hash,
    string_to_number,
    string_upper,
    text_vectorization,
    edit_distance,
    as_string,
    AsStringConfig,
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
