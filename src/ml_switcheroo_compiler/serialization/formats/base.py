"""Define base interfaces for weight formats."""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from collections.abc import Mapping

from ml_switcheroo_compiler.core.shape import OpValue


@typing.runtime_checkable
class WeightFormatProtocol(typing.Protocol):
    """Protocol for weight format handlers supporting both loading and saving."""

    def load(self, filepath: str) -> Mapping[str, OpValue]:
        """Load weights from a file.

        Args:
            filepath (str): Path to the file.

        Returns:
            Mapping[str, OpValue]: The loaded weights dictionary.
        """
        ...

    def save(self, weights_np: Mapping[str, OpValue], filepath: str) -> None:
        """Save weights to a file.

        Args:
            weights_np (Mapping[str, OpValue]): The weights to save.
            filepath (str): Path to the file.
        """
        ...


class WeightLoader(ABC):
    """Interface for loading weights from a file."""

    @abstractmethod
    def load(self, filepath: str) -> Mapping[str, OpValue]:
        """Load weights from a file.

        Args:
            filepath (str): Path to the file.

        Returns:
            Mapping[str, OpValue]: The loaded weights dictionary.
        """
        ...


class WeightSaver(ABC):
    """Interface for saving weights to a file."""

    @abstractmethod
    def save(self, weights_np: Mapping[str, OpValue], filepath: str) -> None:
        """Save weights to a file.

        Args:
            weights_np (Mapping[str, OpValue]): The weights to save.
            filepath (str): Path to the file.
        """
        ...
