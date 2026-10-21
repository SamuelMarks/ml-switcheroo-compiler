"""Pytorch Code Generator Package."""

from __future__ import annotations

import importlib.util
import sys

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_callable

try:
    _has_pkg = importlib.util.find_spec("torch") is not None
except ValueError:
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'pytorch' backend requires the 'torch' library to be installed.")

from .distributed_collectives import (
    pytorch_all_gather,
    pytorch_all_reduce,
    pytorch_broadcast,
    pytorch_reduce_scatter,
)
from .eager import execute_op
from .generator import PyTorchCodeGenerator
from .types import array, asarray, item, zeros

__all__ = [
    "PyTorchCodeGenerator",
    "array",
    "asarray",
    "execute_op",
    "item",
    "pytorch_all_gather",
    "pytorch_all_reduce",
    "pytorch_broadcast",
    "pytorch_reduce_scatter",
    "zeros",
]


def _all_reduce(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Delegate all_reduce to distributed_collectives.pytorch_all_reduce.

    Args:
        *args (EagerValue): Positional arguments.
        **kwargs (EagerValue): Keyword arguments.

    Returns:
        EagerValue: Reduced tensor result.
    """
    import importlib

    dc = importlib.import_module("ml_switcheroo_compiler.backends.pytorch.distributed_collectives")
    return as_callable(getattr(dc, "pytorch_all_reduce"))(*args, **kwargs)


def _all_gather(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Delegate all_gather to distributed_collectives.pytorch_all_gather.

    Args:
        *args (EagerValue): Positional arguments.
        **kwargs (EagerValue): Keyword arguments.

    Returns:
        EagerValue: Gathered tensor result.
    """
    import importlib

    dc = importlib.import_module("ml_switcheroo_compiler.backends.pytorch.distributed_collectives")
    return as_callable(getattr(dc, "pytorch_all_gather"))(*args, **kwargs)


def _reduce_scatter(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Delegate reduce_scatter to distributed_collectives.pytorch_reduce_scatter.

    Args:
        *args (EagerValue): Positional arguments.
        **kwargs (EagerValue): Keyword arguments.

    Returns:
        EagerValue: Scattered tensor result.
    """
    import importlib

    dc = importlib.import_module("ml_switcheroo_compiler.backends.pytorch.distributed_collectives")
    return as_callable(getattr(dc, "pytorch_reduce_scatter"))(*args, **kwargs)


def _broadcast(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Delegate broadcast to distributed_collectives.pytorch_broadcast.

    Args:
        *args (EagerValue): Positional arguments.
        **kwargs (EagerValue): Keyword arguments.

    Returns:
        EagerValue: Broadcasted tensor result.
    """
    import importlib

    dc = importlib.import_module("ml_switcheroo_compiler.backends.pytorch.distributed_collectives")
    return as_callable(getattr(dc, "pytorch_broadcast"))(*args, **kwargs)


type.__setattr__(PyTorchCodeGenerator, "zeros", classmethod(zeros))
type.__setattr__(PyTorchCodeGenerator, "array", classmethod(array))
type.__setattr__(PyTorchCodeGenerator, "asarray", classmethod(asarray))
type.__setattr__(PyTorchCodeGenerator, "item", classmethod(item))
type.__setattr__(PyTorchCodeGenerator, "execute_op", as_callable(classmethod)(execute_op))
type.__setattr__(PyTorchCodeGenerator, "all_reduce", staticmethod(_all_reduce))
type.__setattr__(PyTorchCodeGenerator, "all_gather", staticmethod(_all_gather))
type.__setattr__(PyTorchCodeGenerator, "reduce_scatter", staticmethod(_reduce_scatter))
type.__setattr__(PyTorchCodeGenerator, "broadcast", staticmethod(_broadcast))
