"""I/O and memory operations."""

from __future__ import annotations

import glob  # noqa: F401
import os  # noqa: F401
import shutil  # noqa: F401
import typing

from ml_switcheroo_compiler.core.config import config as core_config
from ml_switcheroo_compiler.core.dtype import DType  # noqa: F401
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor  # noqa: F401
from ml_switcheroo_compiler.ops.base import OpDef, register_op  # noqa: F401
from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat  # noqa: F401
from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat  # noqa: F401
from ml_switcheroo_compiler.serialization.utils import load_npz  # noqa: F401


@typing.runtime_checkable
class _SetDefaultStreamBackendProtocol(typing.Protocol):
    """Protocol for backends exposing `set_default_stream`."""

    def set_default_stream(self, stream: TensorOrScalar) -> None:
        """Set default execution stream.

        Args:
            stream (TensorOrScalar): Stream handle.
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


def set_default_stream(stream: TensorOrScalar) -> None:
    """Set the default stream for the active backend.

    Args:
        stream (TensorOrScalar): The stream parameter.

    Returns:
        NoneType: Result.
    """
    if core_config.backend == "mlx":
        try:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            backend_cls: TensorOrScalar | _SetDefaultStreamBackendProtocol = get_active_backend()
            if isinstance(backend_cls, _SetDefaultStreamBackendProtocol):
                backend_cls.set_default_stream(stream)
        except ImportError:
            return


def set_memory_limit(limit: int) -> None:
    """Set the memory limit for the active backend.

    Args:
        limit (int): The limit parameter.

    Returns:
        NoneType: Result.
    """
    if core_config.backend == "mlx":
        try:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            backend_cls: TensorOrScalar | _SetMemoryLimitBackendProtocol = get_active_backend()
            if isinstance(backend_cls, _SetMemoryLimitBackendProtocol):
                backend_cls.set_memory_limit(limit)
        except ImportError:
            return


def set_wired_limit(limit: int) -> None:
    """Set the wired memory limit for the active backend.

    Args:
        limit (int): The limit parameter.

    Returns:
        NoneType: Result.
    """
    if core_config.backend == "mlx":
        try:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            backend_cls: TensorOrScalar | _SetWiredLimitBackendProtocol = get_active_backend()
            if isinstance(backend_cls, _SetWiredLimitBackendProtocol):
                backend_cls.set_wired_limit(limit)
        except ImportError:
            return
