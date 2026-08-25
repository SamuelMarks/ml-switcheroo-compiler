# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Type mapping utilities for backend generators."""

import typing


class TypeMapper:
    """Handle mapping between generic IR types and backend-specific types."""

    def __init__(self, type_dict: typing.Optional[dict[str, object]] = None) -> None:
        """Initialize.

        Args:
            type_dict (typing.Optional[dict[str, object]]): Dictionary of type mappings.
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
