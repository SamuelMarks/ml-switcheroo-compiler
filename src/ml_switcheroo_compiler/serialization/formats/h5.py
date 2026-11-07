"""H5 format serialization."""

from __future__ import annotations

import typing
from types import ModuleType

from ml_switcheroo_compiler.backends.protocols import has_backend_member
from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.interpreter.evaluator import TensorLike
from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver

h5py: ModuleType | None
try:
    import h5py as _h5py

    h5py = _h5py
except ImportError:
    h5py = None

if typing.TYPE_CHECKING:
    import h5py as h5py_types


class _LoadH5BackendProtocol(typing.Protocol):
    """Protocol for backends implementing custom `load_h5`."""

    def load_h5(self, filepath: str) -> dict[str, TensorLike]:
        """Load HDF5 weights via backend hook.

        Args:
            filepath (str): Path to HDF5 file.

        Returns:
            dict[str, TensorLike]: Loaded weights dictionary.
        """
        ...


class _SaveH5BackendProtocol(typing.Protocol):
    """Protocol for backends implementing custom `save_h5`."""

    def save_h5(self, weights_np: dict[str, TensorLike], filepath: str) -> None:
        """Save HDF5 weights via backend hook.

        Args:
            weights_np (dict[str, TensorLike]): Weights dictionary.
            filepath (str): Destination file path.
        """
        ...


def _is_load_h5_backend(val: TensorOrScalar) -> typing.TypeGuard[_LoadH5BackendProtocol]:
    """Check whether `val` exposes `load_h5`.

    Args:
        val (TensorOrScalar): Active backend.

    Returns:
        bool: True if `val` has `load_h5`.
    """
    return has_backend_member(val, "load_h5")


def _is_save_h5_backend(val: TensorOrScalar) -> typing.TypeGuard[_SaveH5BackendProtocol]:
    """Check whether `val` exposes `save_h5`.

    Args:
        val (TensorOrScalar): Active backend.

    Returns:
        bool: True if `val` has `save_h5`.
    """
    return has_backend_member(val, "save_h5")


@typing.runtime_checkable
class _NumpyConvertibleProtocol(typing.Protocol):
    """Protocol for tensors supporting `.numpy()`."""

    def numpy(self) -> TensorLike:
        """Convert to NumPy array.

        Returns:
            TensorLike: NumPy array.
        """
        ...


@typing.runtime_checkable
class _DataFieldProtocol(typing.Protocol):
    """Protocol for wrapper tensors exposing `.data`."""

    data: TensorLike | _NumpyConvertibleProtocol | _ToListConvertibleProtocol


@typing.runtime_checkable
class _ToListConvertibleProtocol(typing.Protocol):
    """Protocol for array-like objects supporting `.tolist()`."""

    def tolist(self) -> list[int]:
        """Convert to Python list.

        Returns:
            list[int]: Nested Python list.
        """
        ...


class H5WeightFormat(WeightLoader, WeightSaver):
    """H5 weight format handler."""

    def load(self, filepath: str) -> dict[str, TensorLike]:
        """Load h5 weights.

        Args:
            filepath (str): The filepath parameter.

        Returns:
            dict[str, TensorLike]: Loaded weights dictionary.

        Raises:
            ImportError: If h5py is not installed.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend: TensorOrScalar = get_active_backend()
        if _is_load_h5_backend(backend):
            return dict(backend.load_h5(filepath))

        if h5py is None:
            raise ImportError("h5py is required for H5 weight loading. Please install h5py.")

        np_mod = BackendRegistry.get("numpy").get_module()
        result: dict[str, TensorLike] = {}
        with h5py.File(filepath, "r") as f:

            def _visit(name: str, node: h5py_types.Group | h5py_types.Dataset) -> None:
                """Visit h5py items to extract datasets.

                Args:
                    name (str): The dataset name.
                    node (h5py_types.Group | h5py_types.Dataset): The node object.
                """
                if isinstance(node, h5py.Dataset):
                    result[name] = np_mod.asarray(node[()])

            f.visititems(_visit)
        return result

    def save(self, weights_np: dict[str, TensorLike], filepath: str) -> None:
        """Save h5 weights.

        Args:
            weights_np (dict[str, TensorLike]): The weights dictionary to save.
            filepath (str): The destination filepath.

        Raises:
            ImportError: If h5py is not installed.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend: TensorOrScalar = get_active_backend()
        if _is_save_h5_backend(backend):
            backend.save_h5(weights_np, filepath)
            return

        if h5py is None:
            raise ImportError("h5py is required for H5 weight saving. Please install h5py.")

        with h5py.File(filepath, "w") as f:
            from ml_switcheroo_compiler.backends.protocols import ensure_callable, get_backend_member, has_backend_member

            for k, v in weights_np.items():
                val_data: TensorLike | list[int] = v
                if has_backend_member(v, "numpy"):
                    val_data = ensure_callable(get_backend_member(v, "numpy"))()
                elif has_backend_member(v, "data") and has_backend_member(get_backend_member(v, "data"), "numpy"):
                    val_data = ensure_callable(get_backend_member(get_backend_member(v, "data"), "numpy"))()
                elif has_backend_member(v, "tolist"):
                    val_data = ensure_callable(get_backend_member(v, "tolist"))()
                f.create_dataset(k, data=val_data)
            del weights_np
