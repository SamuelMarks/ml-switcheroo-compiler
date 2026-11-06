"""Pytorch Code Generator Package."""

from __future__ import annotations

import importlib.util
import sys
import typing

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_callable
from ml_switcheroo_compiler.backends.protocols import ensure_mod

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
    return ensure_mod(dc).pytorch_all_reduce(*args, **kwargs)


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
    return ensure_mod(dc).pytorch_all_gather(*args, **kwargs)


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
    return ensure_mod(dc).pytorch_reduce_scatter(*args, **kwargs)


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
    return ensure_mod(dc).pytorch_broadcast(*args, **kwargs)


class _PyTorchGeneratorClassProtocol(typing.Protocol):
    """Protocol for attaching PyTorch classmethods and staticmethods to `PyTorchCodeGenerator`."""

    zeros: EagerValue
    array: EagerValue
    asarray: EagerValue
    item: EagerValue
    execute_op: EagerValue
    all_reduce: EagerValue
    all_gather: EagerValue
    reduce_scatter: EagerValue
    broadcast: EagerValue


def _is_pytorch_gen_cls(cls_obj: type) -> typing.TypeGuard[_PyTorchGeneratorClassProtocol]:
    """Type guard narrowing `cls_obj` to `_PyTorchGeneratorClassProtocol`.

    Args:
        cls_obj (type): Target class.

    Returns:
        bool: Always True.
    """
    return cls_obj is not None


_raw_pt_cls: type = PyTorchCodeGenerator
assert _is_pytorch_gen_cls(_raw_pt_cls)
_raw_pt_cls.zeros = ensure_callable(classmethod)(zeros)
_raw_pt_cls.array = ensure_callable(classmethod)(array)
_raw_pt_cls.asarray = ensure_callable(classmethod)(asarray)
_raw_pt_cls.item = ensure_callable(classmethod)(item)
_raw_pt_cls.execute_op = ensure_callable(classmethod)(execute_op)
_raw_pt_cls.all_reduce = ensure_callable(staticmethod)(_all_reduce)
_raw_pt_cls.all_gather = ensure_callable(staticmethod)(_all_gather)
_raw_pt_cls.reduce_scatter = ensure_callable(staticmethod)(_reduce_scatter)
_raw_pt_cls.broadcast = ensure_callable(staticmethod)(_broadcast)
