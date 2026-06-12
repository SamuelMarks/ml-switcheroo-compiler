"""Provides the Environment class for managing variable state and tensor memory mappings.

during evaluation
"""

from __future__ import annotations

from typing import Any


class Environment:
    """Manages variable state, tensor memory mappings, and inputs during interpretation.

    This class acts as a symbol table or memory store, mapping variable names
    to their corresponding values or tensors during the evaluation of an expression
    or execution of a graph

    Attributes:
    memory (dict[str, Any]): The internal storage mapping variable names to their
    values
    """

    def __init__(self, inputs: dict[str, Any] | None = None) -> None:
        """Initialize the instance.

        Args:
            inputs (dict[str, Any]): The inputs parameter
        """
        self.memory: dict[str, Any] = inputs or {}

    def get(self, name: str) -> object:
        """Get.

        Args:
            name (str): The name parameter

        Returns:
            object: The resulting output
        """
        if name not in self.memory:
            msg = f"Missing input value for node '{name}'"
            raise ValueError(msg)
        return self.memory[name]

    def set(self, name: str, value: object) -> None:
        """Set.

        Args:
            name (str): The name parameter
            value (object): The value parameter
        """
        self.memory[name] = value

    def __contains__(self, name: str) -> bool:
        """Check if an item is in the environment."""
        return name in self.memory
