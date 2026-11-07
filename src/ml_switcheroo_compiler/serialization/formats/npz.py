"""NPZ weight archive serialization format handler."""

from __future__ import annotations

import typing
from collections.abc import Mapping

from ml_switcheroo_compiler.backends.protocols import has_backend_member
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.interpreter.evaluator import TensorLike
from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver
from ml_switcheroo_compiler.serialization.utils import parse_npz


class _LoadNpzBackendProtocol(typing.Protocol):
    """Protocol for backends implementing custom `load_npz`."""

    def load_npz(self, filepath: str) -> dict[str, TensorLike]:
        """Load `.npz` weights via backend hook.

        Args:
            filepath (str): Path to `.npz` archive.

        Returns:
            dict[str, TensorLike]: Loaded weights mapping.
        """
        ...


class _SaveNpzBackendProtocol(typing.Protocol):
    """Protocol for backends implementing custom `save_npz`."""

    def save_npz(self, weights_np: TensorOrScalar, filepath: str) -> None:
        """Save `.npz` weights via backend hook.

        Args:
            weights_np (TensorOrScalar): Weights dictionary.
            filepath (str): Destination `.npz` path.
        """
        ...


def _is_load_npz_backend(val: TensorOrScalar) -> typing.TypeGuard[_LoadNpzBackendProtocol]:
    """Check whether `val` exposes `load_npz`.

    Args:
        val (TensorOrScalar): Active backend.

    Returns:
        bool: True if `val` has `load_npz`.
    """
    return has_backend_member(val, "load_npz")


def _is_save_npz_backend(val: TensorOrScalar) -> typing.TypeGuard[_SaveNpzBackendProtocol]:
    """Check whether `val` exposes `save_npz`.

    Args:
        val (TensorOrScalar): Active backend.

    Returns:
        bool: True if `val` has `save_npz`.
    """
    return has_backend_member(val, "save_npz")


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

        backend: TensorOrScalar = registry.get_active_backend()
        if _is_load_npz_backend(backend):
            try:
                return dict(backend.load_npz(filepath))
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

        backend: TensorOrScalar = registry.get_active_backend()
        if _is_save_npz_backend(backend):
            backend.save_npz(weights_np, filepath)
            return
        np_mod = registry.BackendRegistry.get("numpy").get_module()
        assert isinstance(weights_np, Mapping)
        np_mod.savez(filepath, **{str(k): v for k, v in weights_np.items()})
