"""Pickle format serialization."""

import pickle

from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver


class PickleWeightFormat(WeightLoader, WeightSaver):
    """Pickle weight format handler."""

    def load(self, filepath: str) -> dict:
        """Load pickle weights."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def save(self, weights_np: dict, filepath: str) -> None:
        """Save pickle weights."""
        with open(filepath, "wb") as f:
            pickle.dump(weights_np, f)
