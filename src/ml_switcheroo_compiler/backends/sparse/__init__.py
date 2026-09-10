"""Sparse COO Code Generator and Execution Package."""

import importlib.util
import sys

try:
    _has_pkg = importlib.util.find_spec("sparse") is not None
except (ValueError, ModuleNotFoundError):
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'sparse' backend requires the 'sparse' library to be installed.")

from . import eager, generator, kernels, types
from .eager import execute_op
from .generator import SparseGenerator
from .types import COOTensor, array, asarray, item, zeros

SparseGenerator.zeros = classmethod(zeros)
SparseGenerator.array = classmethod(array)
SparseGenerator.asarray = classmethod(asarray)
SparseGenerator.item = classmethod(item)
SparseGenerator.execute_op = classmethod(execute_op)

__all__ = [
    "COOTensor",
    "SparseGenerator",
    "array",
    "asarray",
    "eager",
    "execute_op",
    "generator",
    "item",
    "kernels",
    "types",
    "zeros",
]
