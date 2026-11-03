"""Cupy Code Generator Package."""

from __future__ import annotations

import importlib.util
import sys

from ml_switcheroo_compiler.backends.eager_registry import as_callable
from ml_switcheroo_compiler.core.shape import _obj_set_attr

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

_obj_set_attr(CupyGenerator, "zeros", classmethod(zeros))
_obj_set_attr(CupyGenerator, "array", classmethod(array))
_obj_set_attr(CupyGenerator, "asarray", classmethod(asarray))
_obj_set_attr(CupyGenerator, "item", classmethod(item))
_obj_set_attr(CupyGenerator, "execute_op", as_callable(classmethod)(execute_op))

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
