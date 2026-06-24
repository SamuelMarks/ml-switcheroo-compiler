"""Base interfaces for weight formats."""

from abc import ABC, abstractmethod


class WeightLoader(ABC):
    """Interface for loading weights from a file."""

    @abstractmethod
    def load(self, filepath: str) -> dict:
        """Load weights from a file.

        Args:
            filepath (str): Path to the file.

        Returns:
            dict: The loaded weights.
        """
        ...


class WeightSaver(ABC):
    """Interface for saving weights to a file."""

    @abstractmethod
    def save(self, weights_np: dict, filepath: str) -> None:
        """Save weights to a file.

        Args:
            weights_np (dict): The weights to save.
            filepath (str): Path to the file.
        """
        ...
