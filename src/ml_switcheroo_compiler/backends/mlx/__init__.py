"""Mlx Code Generator Package."""

from __future__ import annotations

import importlib.util
import sys

from ml_switcheroo_compiler.backends.eager_registry import as_callable

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

type.__setattr__(MLXCodeGenerator, "zeros", classmethod(zeros))
type.__setattr__(MLXCodeGenerator, "array", classmethod(array))
type.__setattr__(MLXCodeGenerator, "asarray", classmethod(asarray))
type.__setattr__(MLXCodeGenerator, "item", classmethod(item))
type.__setattr__(MLXCodeGenerator, "execute_op", as_callable(classmethod)(execute_op))

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
