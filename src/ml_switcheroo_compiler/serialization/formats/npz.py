"""NPZ format serialization."""

from ml_switcheroo_compiler.serialization.formats.base import WeightLoader


class NpzWeightFormat(WeightLoader):
    """NPZ weight format handler."""

    def load(self, filepath: str) -> dict:
        """Load npz weights."""
        import numpy as np  # pragma: no cover

        with np.load(filepath) as npz:  # pragma: no cover
            return {k: npz[k] for k in npz.files}  # pragma: no cover
