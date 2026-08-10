from typing import Any

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""NPZ format serialization."""

from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver
from ml_switcheroo_compiler.serialization.utils import parse_npz


class NpzWeightFormat(WeightLoader, WeightSaver):
    """NPZ weight format handler."""

    def load(self, filepath: str) -> dict:
        """Load npz weights.

        Args:
        filepath (str): The filepath parameter.

        Returns:
        dict: Result.
        """
        import ml_switcheroo_compiler.backends.registry as registry

        backend = registry.get_active_backend()
        if hasattr(backend, "load_npz"):
            try:
                return backend.load_npz(filepath)
            except Exception as e:
                import warnings

                warnings.warn(f"Backend load_npz failed: {e}. Falling back to default parse_npz.", stacklevel=2)

        return parse_npz(filepath)

    def save(self, weights_np: dict, filepath: str) -> None:
        """Save npz weights.

        Args:
            weights_np (dict): The weights_np parameter.
            filepath (str): The filepath parameter.

        Returns: Any: Result.
        """
        import numpy as np

        import ml_switcheroo_compiler.backends.registry as registry

        backend = registry.get_active_backend()
        if hasattr(backend, "save_npz"):
            backend.save_npz(weights_np, filepath)
            return
        np.savez(filepath, **weights_np)
