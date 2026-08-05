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

        Returns:
            object: Result.
        """
        import numpy as np

        import ml_switcheroo_compiler.backends.registry as registry

        backend = registry.get_active_backend()
        if hasattr(backend, "save_npz"):
            backend.save_npz(weights_np, filepath)
            return
        np.savez(filepath, **weights_np)
