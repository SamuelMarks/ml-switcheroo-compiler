"""Define base interfaces for weight formats."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class WeightLoader(ABC):
    """Interface for loading weights from a file."""

    @abstractmethod
    def load(self, filepath: str) -> dict[str, np.ndarray]:
        """Load weights from a file.

        Args:
            filepath (str): Path to the file.

        Returns:
            dict[str, np.ndarray]: The loaded weights dictionary.
        """
        ...


class WeightSaver(ABC):
    """Interface for saving weights to a file."""

    @abstractmethod
    def save(self, weights_np: dict[str, np.ndarray], filepath: str) -> None:
        """Save weights to a file.

        Args:
            weights_np (dict[str, np.ndarray]): The weights to save.
            filepath (str): Path to the file.
        """
        ...
