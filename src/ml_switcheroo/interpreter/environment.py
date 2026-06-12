"""Environment for tensor memory mappings and state during evaluation."""

from typing import Any


class Environment:
    """Manages variable state during interpretation."""

    def __init__(self, inputs: dict[str, Any] = None) -> None:
        """Docstring."""
        self.memory: dict[str, Any] = inputs or {}

    def get(self, name: str) -> object:
        """Docstring."""
        if name not in self.memory:
            raise ValueError(f"Missing input value for node '{name}'")
        return self.memory[name]

    def set(self, name: str, value: object) -> None:
        """Docstring."""
        self.memory[name] = value

    def __contains__(self, name: str) -> bool:
        """Check if an item is in the environment."""
        return name in self.memory
