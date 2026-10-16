"""NPZ format serialization."""

from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.interpreter.evaluator import TensorLike
from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver
from ml_switcheroo_compiler.serialization.utils import parse_npz


class NpzWeightFormat(WeightLoader, WeightSaver):
    """NPZ weight format handler."""

    def load(self, filepath: str) -> dict[str, TensorLike]:
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
                return cast("dict[str, TensorLike]", backend.load_npz(filepath))
            except Exception as e:
                import warnings

                warnings.warn(f"Backend load_npz failed: {e}. Falling back to default parse_npz.", stacklevel=2)

        return parse_npz(filepath)

    def save(self, weights_np: EagerValue, filepath: str) -> None:
        """Save npz weights.

        Args:
            weights_np (dict): The weights_np parameter.
            filepath (str): The filepath parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        import ml_switcheroo_compiler.backends.registry as registry

        backend = registry.get_active_backend()
        if hasattr(backend, "save_npz"):
            backend.save_npz(weights_np, filepath)
            return
        np_mod = registry.BackendRegistry.get("numpy").get_module()
        np_mod.savez(filepath, **weights_np)
