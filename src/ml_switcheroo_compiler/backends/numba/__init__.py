"""Numba Code Generator and Execution Package."""

from __future__ import annotations

import importlib.util
import sys
import typing

from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol

try:
    _has_pkg = importlib.util.find_spec("numba") is not None
except (ValueError, ModuleNotFoundError):
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'numba' backend requires the 'numba' library to be installed.")

from .eager import execute_op
from .generator import NumbaGenerator
from .types import array, asarray, item, zeros

type.__setattr__(NumbaGenerator, "zeros", staticmethod(zeros))
type.__setattr__(NumbaGenerator, "array", staticmethod(array))
type.__setattr__(NumbaGenerator, "asarray", staticmethod(asarray))
type.__setattr__(NumbaGenerator, "item", staticmethod(item))
type.__setattr__(NumbaGenerator, "execute_op", typing.cast("EagerCallableProtocol", staticmethod)(execute_op))

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
