"""Cupy Code Generator Package."""

from __future__ import annotations

import importlib.util
import sys
import typing

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_callable

try:
    _has_pkg = importlib.util.find_spec("cupy") is not None
except ValueError:
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'cupy' backend requires the 'cupy' library to be installed.")

from . import eager, generator, types
from .eager import execute_op
from .generator import CupyGenerator
from .types import array, asarray, item, zeros


class _CupyGeneratorClassProtocol(typing.Protocol):
    """Protocol for attaching CuPy classmethods to `CupyGenerator`."""

    zeros: EagerValue
    array: EagerValue
    asarray: EagerValue
    item: EagerValue
    execute_op: EagerValue


def _is_cupy_gen_cls(cls_obj: type) -> typing.TypeGuard[_CupyGeneratorClassProtocol]:
    """Type guard narrowing `cls_obj` to `_CupyGeneratorClassProtocol`.

    Args:
        cls_obj (type): Target class.

    Returns:
        bool: Always True.
    """
    return cls_obj is not None


_raw_cupy_cls: type = CupyGenerator
assert _is_cupy_gen_cls(_raw_cupy_cls)
_raw_cupy_cls.zeros = ensure_callable(classmethod)(zeros)
_raw_cupy_cls.array = ensure_callable(classmethod)(array)
_raw_cupy_cls.asarray = ensure_callable(classmethod)(asarray)
_raw_cupy_cls.item = ensure_callable(classmethod)(item)
_raw_cupy_cls.execute_op = ensure_callable(classmethod)(execute_op)

__all__ = [
    "CupyGenerator",
    "array",
    "asarray",
    "eager",
    "execute_op",
    "generator",
    "item",
    "types",
    "zeros",
]
