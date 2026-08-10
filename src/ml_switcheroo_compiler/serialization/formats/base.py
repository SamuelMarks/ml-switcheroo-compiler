# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Define base interfaces for weight formats."""

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
        pass


class WeightSaver(ABC):
    """Interface for saving weights to a file."""

    @abstractmethod
    def save(self, weights_np: dict, filepath: str) -> None:
        """Save weights to a file.

        Args:
            weights_np (dict): The weights to save.
            filepath (str): Path to the file.
        """
        pass
