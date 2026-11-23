"""NPZ weight archive serialization format handler."""

from __future__ import annotations

import typing
from collections.abc import Mapping

from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver
from ml_switcheroo_compiler.serialization.utils import _tensor_to_npy_bytes, parse_npz


@typing.runtime_checkable
class _LoadNpzBackendProtocol(typing.Protocol):
    """Protocol for backends implementing custom `load_npz`."""

    def load_npz(self, filepath: str) -> dict[str, OpValue]:
        """Load `.npz` weights via backend hook.

        Args:
            filepath (str): Path to `.npz` archive.

        Returns:
            dict[str, OpValue]: Loaded weights mapping.
        """
        ...


@typing.runtime_checkable
class _SaveNpzBackendProtocol(typing.Protocol):
    """Protocol for backends implementing custom `save_npz`."""

    def save_npz(self, weights_np: OpValue, filepath: str) -> None:
        """Save `.npz` weights via backend hook.

        Args:
            weights_np (OpValue): Weights dictionary.
            filepath (str): Destination `.npz` path.
        """
        ...


def _is_load_npz_backend(val: OpValue) -> typing.TypeGuard[_LoadNpzBackendProtocol]:
    """Check whether `val` exposes `load_npz`.

    Args:
        val (OpValue): Active backend.

    Returns:
        bool: True if `val` has `load_npz`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _LoadNpzBackendProtocol)
    return found or is_mock_with_attrs(val, "load_npz")


def _is_save_npz_backend(val: OpValue) -> typing.TypeGuard[_SaveNpzBackendProtocol]:
    """Check whether `val` exposes `save_npz`.

    Args:
        val (OpValue): Active backend.

    Returns:
        bool: True if `val` has `save_npz`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _SaveNpzBackendProtocol)
    return found or is_mock_with_attrs(val, "save_npz")


class NpzWeightFormat(WeightLoader, WeightSaver):
    """Weight loader and saver for `.npz` zip archives of `.npy` arrays."""

    def load(self, filepath: str) -> dict[str, OpValue]:
        """Load a dictionary of named weight arrays from an `.npz` file.

        Args:
            filepath (str): Source `.npz` file path.

        Returns:
            dict[str, OpValue]: Mapping of weight names to loaded arrays.
        """
        import ml_switcheroo_compiler.backends.registry as registry

        backend: OpValue = registry.get_active_backend()
        if _is_load_npz_backend(backend):
            try:
                return dict(backend.load_npz(filepath))
            except (
                EOFError,
                ImportError,
                ModuleNotFoundError,
                OSError,
                AttributeError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as e:
                import warnings

                warnings.warn(
                    f"Backend load_npz failed: {e}. Falling back to default parse_npz.",
                    stacklevel=2,
                )

        return parse_npz(filepath)

    def save(self, weights_np: OpValue, filepath: str) -> None:
        """Save a dictionary of named weight arrays to an `.npz` file.

        Args:
            weights_np (OpValue): Mapping of weight names to arrays.
            filepath (str): Destination `.npz` file path.
        """
        import ml_switcheroo_compiler.backends.registry as registry

        backend: OpValue = registry.get_active_backend()
        if _is_save_npz_backend(backend):
            backend.save_npz(weights_np, filepath)
            return
        import zipfile

        assert isinstance(weights_np, Mapping)
        with zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_STORED) as zf:
            for k, v in weights_np.items():
                k_str = str(k)
                entry_name = k_str if k_str.endswith(".npy") else f"{k_str}.npy"
                zf.writestr(entry_name, _tensor_to_npy_bytes(v))
