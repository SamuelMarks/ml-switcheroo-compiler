"""NPZ weight archive serialization format handler."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.interpreter.evaluator import TensorLike
from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver
from ml_switcheroo_compiler.serialization.utils import parse_npz


class NpzWeightFormat(WeightLoader, WeightSaver):
    """Weight loader and saver for `.npz` zip archives of `.npy` arrays."""

    def load(self, filepath: str) -> dict[str, TensorLike]:
        """Load a dictionary of named weight arrays from an `.npz` file.

        Args:
            filepath (str): Path to the `.npz` archive file.

        Returns:
            dict[str, TensorLike]: Mapping of weight names to loaded arrays.
        """
        import ml_switcheroo_compiler.backends.registry as registry

        backend = registry.get_active_backend()
        if hasattr(backend, "load_npz"):
            try:
                return typing.cast("dict[str, TensorLike]", backend.load_npz(filepath))
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Backend load_npz failed: {e}. Falling back to default parse_npz.",
                    stacklevel=2,
                )

        return parse_npz(filepath)

    def save(self, weights_np: TensorOrScalar, filepath: str) -> None:
        """Save a dictionary of named weight arrays to an `.npz` file.

        Args:
            weights_np (TensorOrScalar): Mapping of weight names to arrays.
            filepath (str): Destination `.npz` file path.
        """
        import ml_switcheroo_compiler.backends.registry as registry

        backend = registry.get_active_backend()
        if hasattr(backend, "save_npz"):
            backend.save_npz(weights_np, filepath)
            return
        np_mod = registry.BackendRegistry.get("numpy").get_module()
        np_mod.savez(filepath, **typing.cast("dict[str, TensorOrScalar]", weights_np))
