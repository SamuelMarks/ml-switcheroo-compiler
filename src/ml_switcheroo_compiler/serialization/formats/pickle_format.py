"""Pickle format serialization."""

from __future__ import annotations

import pickle
import typing

from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.interpreter.evaluator import TensorLike
from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver


class PickleWeightFormat(WeightLoader, WeightSaver):
    """Pickle weight format handler."""

    def load(self, filepath: str) -> dict[str, TensorLike]:
        """Load pickle weights.

        Args:
            filepath (str): The filepath parameter.

        Returns:
            dict[str, TensorLike]: Loaded weight dictionary.
        """
        with open(filepath, "rb") as f:
            return typing.cast(dict[str, TensorLike], pickle.load(f))

    def save(self, weights_np: TensorOrScalar, filepath: str) -> None:
        """Save pickle weights.

        Args:
            weights_np (TensorOrScalar): The weights_np parameter.
            filepath (str): The filepath parameter.
        """
        with open(filepath, "wb") as f:
            pickle.dump(weights_np, f)
