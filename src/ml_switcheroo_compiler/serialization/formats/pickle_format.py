"""Pickle format serialization."""

from __future__ import annotations

import pickle
import typing

from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.interpreter.evaluator import TensorLike
from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver


def _is_weight_dict(val: TensorOrScalar) -> typing.TypeGuard[dict[str, TensorLike]]:
    """Type guard narrowing `val` to `dict[str, TensorLike]`.

    Args:
        val (TensorOrScalar): Candidate deserialized object.

    Returns:
        bool: True if val is a dictionary.
    """
    return isinstance(val, dict)


class PickleWeightFormat(WeightLoader, WeightSaver):
    """Pickle weight format handler."""

    def load(self, filepath: str) -> dict[str, TensorLike]:
        """Load pickle weights.

        Args:
            filepath (str): Input `filepath` argument (filepath).

        Returns:
            dict[str, TensorLike]: Loaded weight dictionary.
        """
        with open(filepath, "rb") as f:
            loaded: TensorOrScalar = pickle.load(f)
            assert _is_weight_dict(loaded)
            return loaded

    def save(self, weights_np: TensorOrScalar, filepath: str) -> None:
        """Save pickle weights.

        Args:
            weights_np (TensorOrScalar): Input `weights_np` argument (weights np).
            filepath (str): Input `filepath` argument (filepath).
        """
        with open(filepath, "wb") as f:
            pickle.dump(weights_np, f)
