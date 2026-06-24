"""Type mapping utilities for backend generators."""

from typing import Any  # pragma: no cover


class TypeMapper:  # pragma: no cover
    """Handles mapping between generic IR types and backend-specific types."""

    def __init__(self, type_dict: dict[str, Any] | None = None) -> None:  # noqa: ANN401  # pragma: no cover
        """Initialize.

        Args:
            type_dict (dict[str, Any] | None): Dictionary of type mappings.
        """
        self.type_dict = type_dict or {}  # pragma: no cover

    def map_type(self, ir_type: str) -> str:  # pragma: no cover
        """Map generic IR type to target type.

        Args:
            ir_type (str): The IR type string.

        Returns:
            str: The target backend type string.
        """
        return self.type_dict.get(ir_type, ir_type)  # pragma: no cover
