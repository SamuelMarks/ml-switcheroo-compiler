"""Provide the Environment class for managing variable state and tensor memory mappings during evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor

ValueType = Union["Tensor", float, int, bool, str, tuple[int, ...], list[float]]


class Environment:
    """Manages variable state, tensor memory mappings, and inputs during interpretation.

    This class acts as a symbol table or memory store, mapping variable names
    to their corresponding values or tensors during the evaluation of an expression
    or execution of a graph.

    Attributes:
        memory (dict[str, ValueType]): The internal storage mapping variable names to their values.
    """

    memory: dict[str, ValueType]

    def __init__(self, inputs: dict[str, ValueType] | None = None) -> None:
        """Initialize the environment.

        Args:
            inputs (dict[str, ValueType] | None): Initial variable mapping dictionary.
        """
        self.memory = inputs if inputs is not None else {}

    def get(self, name: str) -> ValueType:
        """Retrieve the value associated with the given node or variable name.

        Args:
            name (str): The unique identifier for the tensor or variable to retrieve.

        Returns:
            ValueType: The concrete tensor, scalar, or value associated with the name.

        Raises:
            ValueError: If the requested name does not exist in the environment's memory.
        """
        if name not in self.memory:
            msg = f"Missing input value for node '{name}'"
            raise ValueError(msg)
        return self.memory[name]

    def set(self, name: str, value: ValueType) -> None:
        """Store or update a value in the environment for a specific node or variable.

        Args:
            name (str): The unique identifier where the value should be stored.
            value (ValueType): The concrete tensor, scalar, or object to store.
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
