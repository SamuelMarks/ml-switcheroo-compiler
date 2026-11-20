"""Pickle format serialization."""

from __future__ import annotations

import pickle

from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.interpreter.environment import InterpreterValue, as_eval_value
from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver


class PickleWeightFormat(WeightLoader, WeightSaver):
    """Pickle weight format handler."""

    def load(self, filepath: str) -> dict[str, InterpreterValue]:
        """Load pickle weights.

        Args:
            filepath (str): Input `filepath` argument (filepath).

        Returns:
            dict[str, InterpreterValue]: Loaded weight dictionary.

        Raises:
            TypeError: If deserialized payload is not a dictionary.
        """
        with open(filepath, "rb") as f:
            loaded: OpValue = pickle.load(f)
            if not isinstance(loaded, dict):
                raise TypeError(f"Expected dict weight payload, got {type(loaded).__name__}")
            return {str(k): as_eval_value(v) for k, v in loaded.items()}

    def save(self, weights_np: OpValue, filepath: str) -> None:
        """Save pickle weights.

        Args:
            weights_np (OpValue): Input `weights_np` argument (weights np).
            filepath (str): Input `filepath` argument (filepath).
        """
        with open(filepath, "wb") as f:
            pickle.dump(weights_np, f)
