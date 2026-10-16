"""Define base interfaces for weight formats."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ml_switcheroo_compiler.interpreter.evaluator import TensorLike


class WeightLoader(ABC):
    """Interface for loading weights from a file."""

    @abstractmethod
    def load(self, filepath: str) -> dict[str, TensorLike]:
        """Load weights from a file.

        Args:
            filepath (str): Path to the file.

        Returns:
            dict[str, TensorLike]: The loaded weights dictionary.
        """
        ...


class WeightSaver(ABC):
    """Interface for saving weights to a file."""

    @abstractmethod
    def save(self, weights_np: dict[str, TensorLike], filepath: str) -> None:
        """Save weights to a file.

        Args:
            weights_np (dict[str, TensorLike]): The weights to save.
            filepath (str): Path to the file.
        """
        ...
