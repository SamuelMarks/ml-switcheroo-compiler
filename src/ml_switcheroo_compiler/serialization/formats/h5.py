"""Consolidated validated HDF5 weight serialization format."""

from __future__ import annotations

import typing
from collections.abc import Mapping
from types import ModuleType

from pydantic import BaseModel, ConfigDict, Field

from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver

h5py: ModuleType | None
try:
    import h5py as _h5py

    h5py = _h5py
except ImportError:
    h5py = None

if typing.TYPE_CHECKING:
    import h5py as h5py_types


class WeightSchema(BaseModel):
    """Pydantic schema validating HDF5 weight dictionaries."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: dict[str, OpValue] = Field(description="Mapping from tensor names to weight arrays")


@typing.runtime_checkable
class _LoadH5BackendProtocol(typing.Protocol):
    """Protocol for backends implementing custom ``load_h5``."""

    def load_h5(self, filepath: str) -> dict[str, OpValue]:
        """Load HDF5 weights via backend hook.

        Args:
            filepath (str): Path to HDF5 file.

        Returns:
            dict[str, OpValue]: Loaded weights dictionary.
        """
        ...


@typing.runtime_checkable
class _SaveH5BackendProtocol(typing.Protocol):
    """Protocol for backends implementing custom ``save_h5``."""

    def save_h5(self, weights_np: Mapping[str, OpValue], filepath: str) -> None:
        """Save HDF5 weights via backend hook.

        Args:
            weights_np (Mapping[str, OpValue]): Weights dictionary.
            filepath (str): Destination file path.
        """
        ...


@typing.runtime_checkable
class _H5NumpyMethodProto(typing.Protocol):
    """Protocol for tensors exposing a `.numpy()` conversion method."""

    def numpy(self) -> OpValue:
        """Return NumPy array representation.

        Returns:
            OpValue: Converted host array.
        """
        ...


@typing.runtime_checkable
class _H5DataHolderProto(typing.Protocol):
    """Protocol for tensor wrappers exposing a `.data` payload."""

    @property
    def data(self) -> OpValue:
        """Return underlying tensor data buffer.

        Returns:
            OpValue: Backing buffer.
        """
        ...


@typing.runtime_checkable
class _H5ToListProto(typing.Protocol):
    """Protocol for arrays exposing `.tolist()`."""

    def tolist(self) -> OpValue:
        """Convert array into nested Python list.

        Returns:
            OpValue: Nested list representation.
        """
        ...


def _is_load_h5_backend(val: OpValue) -> typing.TypeGuard[_LoadH5BackendProtocol]:
    """Check whether ``val`` exposes ``load_h5``.

    Args:
        val (OpValue): Active backend.

    Returns:
        bool: True if ``val`` exposes ``load_h5``.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _LoadH5BackendProtocol)
    return found or is_mock_with_attrs(val, "load_h5")


def _is_save_h5_backend(val: OpValue) -> typing.TypeGuard[_SaveH5BackendProtocol]:
    """Check whether ``val`` exposes ``save_h5``.

    Args:
        val (OpValue): Active backend.

    Returns:
        bool: True if ``val`` exposes ``save_h5``.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _SaveH5BackendProtocol)
    return found or is_mock_with_attrs(val, "save_h5")


def _extract_serializable_array(val: OpValue) -> OpValue:
    """Extract raw array/list payload suitable for ``h5py.File.create_dataset``.

    Args:
        val (OpValue): Weight tensor or array wrapper.

    Returns:
        OpValue: Underlying buffer or nested list.
    """
    probe: OpValue | _H5NumpyMethodProto | _H5DataHolderProto | _H5ToListProto = val
    if type(probe) is not ModuleType and isinstance(probe, _H5NumpyMethodProto):
        return probe.numpy()
    if isinstance(probe, _H5DataHolderProto):
        inner: OpValue | _H5NumpyMethodProto = probe.data
        if type(inner) is not ModuleType and isinstance(inner, _H5NumpyMethodProto):
            return inner.numpy()
    if type(probe) is not ModuleType and isinstance(probe, _H5ToListProto):
        return probe.tolist()
    return val


class H5WeightFormat(WeightLoader, WeightSaver):
    """Canonical HDF5 weight format handler with optional backend hook delegation."""

    def load(self, filepath: str) -> dict[str, OpValue]:
        """Load HDF5 weights from ``filepath``.

        Args:
            filepath (str): Source HDF5 file path.

        Returns:
            dict[str, OpValue]: Validated dictionary of loaded weights.

        Raises:
            ImportError: If ``h5py`` is not installed.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend: OpValue = get_active_backend()
        if _is_load_h5_backend(backend):
            return dict(backend.load_h5(filepath))

        if h5py is None:
            raise ImportError("h5py is required for HDF5 weight loading. Please install h5py.")

        result: dict[str, OpValue] = {}
        with h5py.File(filepath, "r") as f:

            def _visit(name: str, node: h5py_types.Group | h5py_types.Dataset) -> None:
                """Extract dataset leaves during recursive HDF5 traversal.

                Args:
                    name (str): Dataset path inside the HDF5 archive.
                    node (h5py_types.Group | h5py_types.Dataset): Visited HDF5 object.
                """
                if isinstance(node, h5py.Dataset):
                    result[name] = node[()]

            f.visititems(_visit)
        return WeightSchema(data=result).data

    def save(self, weights_np: Mapping[str, OpValue], filepath: str) -> None:
        """Save HDF5 weights to ``filepath``.

        Args:
            weights_np (Mapping[str, OpValue]): Weights dictionary to persist.
            filepath (str): Destination HDF5 file path.

        Raises:
            ImportError: If ``h5py`` is not installed.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend: OpValue = get_active_backend()
        if _is_save_h5_backend(backend):
            backend.save_h5(weights_np, filepath)
            return

        if h5py is None:
            raise ImportError("h5py is required for HDF5 weight saving. Please install h5py.")

        validated = WeightSchema(data=dict(weights_np))
        with h5py.File(filepath, "w") as f:
            for key, val in validated.data.items():
                f.create_dataset(key, data=_extract_serializable_array(val))


class HDF5WeightLoader(H5WeightFormat):
    """Schema-validated HDF5 weight loader."""


class HDF5WeightSaver(H5WeightFormat):
    """Schema-validated HDF5 weight saver."""


__all__ = [
    "H5WeightFormat",
    "HDF5WeightLoader",
    "HDF5WeightSaver",
    "WeightSchema",
]
