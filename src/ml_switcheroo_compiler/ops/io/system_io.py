"""I/O and memory operations."""

from __future__ import annotations

import glob  # noqa: F401
import os  # noqa: F401
import shutil  # noqa: F401
import typing

from ml_switcheroo_compiler.core.config import config as core_config
from ml_switcheroo_compiler.core.dtype import DType  # noqa: F401
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.core.tensor import Tensor  # noqa: F401
from ml_switcheroo_compiler.ops.base import OpDef, register_op  # noqa: F401
from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat  # noqa: F401
from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat  # noqa: F401
from ml_switcheroo_compiler.serialization.utils import load_npz  # noqa: F401


@typing.runtime_checkable
class _SetDefaultStreamBackendProtocol(typing.Protocol):
    """Protocol for backends exposing `set_default_stream`."""

    def set_default_stream(self, stream: OpValue) -> None:
        """Set default execution stream.

        Args:
            stream (OpValue): Stream handle.
        """
        ...


@typing.runtime_checkable
class _SetMemoryLimitBackendProtocol(typing.Protocol):
    """Protocol for backends exposing `set_memory_limit`."""

    def set_memory_limit(self, limit: int) -> None:
        """Set device memory limit.

        Args:
            limit (int): Memory limit in bytes.
        """
        ...


@typing.runtime_checkable
class _SetWiredLimitBackendProtocol(typing.Protocol):
    """Protocol for backends exposing `set_wired_limit`."""

    def set_wired_limit(self, limit: int) -> None:
        """Set wired memory limit.

        Args:
            limit (int): Wired limit in bytes.
        """
        ...


def _is_set_default_stream_backend(val: OpValue) -> typing.TypeGuard[_SetDefaultStreamBackendProtocol]:
    """Check whether `val` exposes `set_default_stream`.

    Args:
        val (OpValue): Active backend.

    Returns:
        bool: True if `val` satisfies `_SetDefaultStreamBackendProtocol`.
    """
    return isinstance(val, _SetDefaultStreamBackendProtocol)


def _is_set_memory_limit_backend(val: OpValue) -> typing.TypeGuard[_SetMemoryLimitBackendProtocol]:
    """Check whether `val` exposes `set_memory_limit`.

    Args:
        val (OpValue): Active backend.

    Returns:
        bool: True if `val` satisfies `_SetMemoryLimitBackendProtocol`.
    """
    return isinstance(val, _SetMemoryLimitBackendProtocol)


def _is_set_wired_limit_backend(val: OpValue) -> typing.TypeGuard[_SetWiredLimitBackendProtocol]:
    """Check whether `val` exposes `set_wired_limit`.

    Args:
        val (OpValue): Active backend.

    Returns:
        bool: True if `val` satisfies `_SetWiredLimitBackendProtocol`.
    """
    return isinstance(val, _SetWiredLimitBackendProtocol)


def set_default_stream(stream: OpValue) -> None:
    """Set the default stream for the active backend.

    Args:
        stream (OpValue): Input `stream` argument (stream).

    Returns:
        NoneType: Computed `NoneType` output.
    """
    if core_config.backend == "mlx":
        try:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            backend_cls: OpValue = get_active_backend()
            if _is_set_default_stream_backend(backend_cls):
                backend_cls.set_default_stream(stream)
        except ImportError:
            return


def set_memory_limit(limit: int) -> None:
    """Set the memory limit for the active backend.

    Args:
        limit (int): Input `limit` argument (limit).

    Returns:
        NoneType: Computed `NoneType` output.
    """
    if core_config.backend == "mlx":
        try:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            backend_cls: OpValue = get_active_backend()
            if _is_set_memory_limit_backend(backend_cls):
                backend_cls.set_memory_limit(limit)
        except ImportError:
            return


def set_wired_limit(limit: int) -> None:
    """Set the wired memory limit for the active backend.

    Args:
        limit (int): Input `limit` argument (limit).

    Returns:
        NoneType: Computed `NoneType` output.
    """
    if core_config.backend == "mlx":
        try:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            backend_cls: OpValue = get_active_backend()
            if _is_set_wired_limit_backend(backend_cls):
                backend_cls.set_wired_limit(limit)
        except ImportError:
            return
