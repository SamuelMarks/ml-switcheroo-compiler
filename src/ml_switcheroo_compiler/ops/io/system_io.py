# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""I/O and memory operations."""

from __future__ import annotations

import glob
import os
import shutil

from ml_switcheroo_compiler.core.config import config as core_config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat
from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat
from ml_switcheroo_compiler.serialization.utils import load_npz


def set_default_stream(stream) -> None:
    """Set the default stream for the active backend.

    Args:
        stream (Any): The stream parameter.

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
