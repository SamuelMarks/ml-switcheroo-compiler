# ruff: noqa: E402, E501, F403
"""Type mapping utilities for backend generators."""

from __future__ import annotations


class TypeMapper:
    """Handle mapping between generic IR types and backend-specific types."""

    type_dict: dict[str, str]

    def __init__(self, type_dict: dict[str, str] | None = None) -> None:
        """Initialize.

        Args:
            type_dict (dict[str, str] | None): Dictionary of type mappings.
        """
        self.type_dict = type_dict or {}

    def map_type(self, ir_type: str) -> str:
        """Map generic IR type to target type.

        Args:
            ir_type (str): The IR type string.

        Returns:
            str: The target backend type string.
        """
        return self.type_dict.get(ir_type, ir_type)
