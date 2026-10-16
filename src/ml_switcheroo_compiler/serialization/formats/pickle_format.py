"""Pickle format serialization."""

from __future__ import annotations

import pickle
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.interpreter.evaluator import TensorLike
from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver


class PickleWeightFormat(WeightLoader, WeightSaver):
    """Pickle weight format handler."""

    def load(self, filepath: str) -> dict[str, TensorLike]:
        """Load pickle weights.

        Args:
        filepath (str): The filepath parameter.

        Returns:
        dict: Result.
        """
        with open(filepath, "rb") as f:
            return cast(dict[str, TensorLike], pickle.load(f))

    def save(self, weights_np: EagerValue, filepath: str) -> None:
        """Save pickle weights.

        Args:
            weights_np (dict): The weights_np parameter.
            filepath (str): The filepath parameter.
        """
        with open(filepath, "wb") as f:
            pickle.dump(weights_np, f)
