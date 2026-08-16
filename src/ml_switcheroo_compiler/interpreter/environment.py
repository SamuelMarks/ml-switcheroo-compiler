"""Module environment.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Provide the Environment class for managing variable state and tensor memory mappings.

during evaluation
"""
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
        """Initialize the object.

        Args:
            inputs (dict[str, Any]): The inputs to process.
        """
        self.memory: dict[str, Any] = inputs or {}

    def get(self, name: str) -> Any:
        """Retrieve the value associated with the given node or variable name.

        Args:
            name (str): The unique identifier for the tensor or variable to retrieve.

        Returns: Any: The concrete tensor, scalar, or value associated with the name.

        Raises:
            ValueError: If the requested name does not exist in the environment's memory.
        """
        if name not in self.memory:
            msg = f"Missing input value for node '{name}'"
            raise ValueError(msg)
        return self.memory[name]

    def set(self, name: str, value: Any) -> None:
        """Store or update a value in the environment for a specific node or variable.

        Args:
            name (str): The unique identifier where the value should be stored.
            value (object): The concrete tensor, scalar, or object to store.
        """
        self.memory[name] = value

    def __contains__(self, name: str) -> bool:
        """Check if a specific node or variable name exists within the environment.

        Args:
            name (str): The unique identifier to check for in the memory store.

        Returns:
            bool: True if the name is present in the environment's memory, False otherwise.
        """
        return name in self.memory
