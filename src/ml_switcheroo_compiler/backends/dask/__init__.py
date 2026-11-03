"""Dask Code Generator Package."""

from __future__ import annotations

import importlib.util
import sys
import typing

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_callable
from ml_switcheroo_compiler.core.shape import StructuralOperandProtocol

try:
    _has_pkg = importlib.util.find_spec("dask") is not None
except ValueError:
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'dask' backend requires the 'dask' library to be installed.")

from .eager import execute_op
from .generator import DaskGenerator
from .types import array, asarray, item, zeros


class _DaskGeneratorClassProtocol(typing.Protocol):
    """Protocol for attaching Dask classmethods to `DaskGenerator`."""

    zeros: EagerValue
    array: EagerValue
    asarray: EagerValue
    item: EagerValue
    execute_op: EagerValue


def _is_dask_gen_cls(cls_obj: StructuralOperandProtocol) -> typing.TypeGuard[_DaskGeneratorClassProtocol]:
    """Type guard narrowing `cls_obj` to `_DaskGeneratorClassProtocol`.

    Args:
        cls_obj (StructuralOperandProtocol): Target class.

    Returns:
        bool: Always True.
    """
    return cls_obj is not None


_raw_dask_cls: StructuralOperandProtocol = DaskGenerator
assert _is_dask_gen_cls(_raw_dask_cls)
_raw_dask_cls.zeros = as_callable(classmethod)(zeros)
_raw_dask_cls.array = as_callable(classmethod)(array)
_raw_dask_cls.asarray = as_callable(classmethod)(asarray)
_raw_dask_cls.item = as_callable(classmethod)(item)
_raw_dask_cls.execute_op = as_callable(classmethod)(execute_op)
