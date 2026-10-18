"""I/O and memory operations."""

from __future__ import annotations

import glob  # noqa: F401
import os  # noqa: F401
import shutil  # noqa: F401

from ml_switcheroo_compiler.core.config import config as core_config
from ml_switcheroo_compiler.core.dtype import DType  # noqa: F401
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor  # noqa: F401
from ml_switcheroo_compiler.ops.base import OpDef, register_op  # noqa: F401
from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat  # noqa: F401
from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat  # noqa: F401
from ml_switcheroo_compiler.serialization.utils import load_npz  # noqa: F401


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

            backend_cls = get_active_backend()
            if hasattr(backend_cls, "set_default_stream"):
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

            backend_cls = get_active_backend()
            if hasattr(backend_cls, "set_memory_limit"):
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

            backend_cls = get_active_backend()
            if hasattr(backend_cls, "set_wired_limit"):
                backend_cls.set_wired_limit(limit)
        except ImportError:
            return
