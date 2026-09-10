"""Cupy Code Generator Package."""

import importlib.util
import sys

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

CupyGenerator.zeros = classmethod(zeros)
CupyGenerator.array = classmethod(array)
CupyGenerator.asarray = classmethod(asarray)
CupyGenerator.item = classmethod(item)
CupyGenerator.execute_op = classmethod(execute_op)

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
