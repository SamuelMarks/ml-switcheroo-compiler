# ruff: noqa: E402, E501, F403
"""Mlx Code Generator Package."""

import importlib.util
import sys

try:
    _has_pkg = importlib.util.find_spec("mlx") is not None
except ValueError:
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'mlx' backend requires the 'mlx' library to be installed.")

from . import eager, generator, profiler, types
from .eager import execute_op
from .generator import MLXCodeGenerator
from .types import array, asarray, item, zeros

MLXCodeGenerator.zeros = classmethod(zeros)
MLXCodeGenerator.array = classmethod(array)
MLXCodeGenerator.asarray = classmethod(asarray)
MLXCodeGenerator.item = classmethod(item)
MLXCodeGenerator.execute_op = classmethod(execute_op)

__all__ = [
    "MLXCodeGenerator",
    "array",
    "asarray",
    "eager",
    "execute_op",
    "generator",
    "item",
    "profiler",
    "types",
    "zeros",
]
