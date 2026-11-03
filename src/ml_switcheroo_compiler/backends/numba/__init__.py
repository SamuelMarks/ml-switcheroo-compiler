"""Numba Code Generator and Execution Package."""

from __future__ import annotations

import importlib.util
import sys

from ml_switcheroo_compiler.backends.eager_registry import as_callable
from ml_switcheroo_compiler.core.shape import _obj_set_attr

try:
    _has_pkg = importlib.util.find_spec("numba") is not None
except (ValueError, ModuleNotFoundError):
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'numba' backend requires the 'numba' library to be installed.")

from .eager import execute_op
from .generator import NumbaGenerator
from .types import array, asarray, item, zeros

_obj_set_attr(NumbaGenerator, "zeros", staticmethod(zeros))
_obj_set_attr(NumbaGenerator, "array", staticmethod(array))
_obj_set_attr(NumbaGenerator, "asarray", staticmethod(asarray))
_obj_set_attr(NumbaGenerator, "item", staticmethod(item))
_obj_set_attr(NumbaGenerator, "execute_op", as_callable(staticmethod)(execute_op))

__all__ = [
    "NumbaGenerator",
    "array",
    "asarray",
    "eager",
    "execute_op",
    "generator",
    "item",
    "types",
    "zeros",
]
