"""Numba Code Generator and Execution Package."""

import importlib.util
import sys

try:
    _has_pkg = importlib.util.find_spec("numba") is not None
except (ValueError, ModuleNotFoundError):
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'numba' backend requires the 'numba' library to be installed.")

from .eager import execute_op
from .generator import NumbaGenerator
from .types import array, asarray, item, zeros

NumbaGenerator.zeros = classmethod(zeros)
NumbaGenerator.array = classmethod(array)
NumbaGenerator.asarray = classmethod(asarray)
NumbaGenerator.item = classmethod(item)
NumbaGenerator.execute_op = classmethod(execute_op)

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
