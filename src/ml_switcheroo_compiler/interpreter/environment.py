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
        """Initializes the object.

        Args:
            inputs (dict[str, Any]): The inputs to process.
        """
        self.memory: dict[str, Any] = inputs or {}

    def get(self, name: str) -> object:
        """Retrieves the value associated with the given name.

        Args:
            name (str): The variable name.

        Returns:
            The computed shape or evaluation result.
        """
        if name not in self.memory:
            msg = f"Missing input value for node '{name}'"
            raise ValueError(msg)
        return self.memory[name]

    def set(self, name: str, value: object) -> None:
        """Sets the value associated with the given name.

        Args:
            name (str): The variable name.
            value (object): The value to set or add.
        """
        self.memory[name] = value

    def __contains__(self, name: str) -> bool:
        """Check if an item is in the environment.

        Args:
            name (str): The name parameter for the operation.

        Returns:
            bool: A boolean indicating the result of the check.
        """
        return name in self.memory
