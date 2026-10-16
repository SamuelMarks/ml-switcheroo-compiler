"""Cupy Code Generator Package."""

import importlib.util
import sys
from typing import Callable, cast

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol

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

type.__setattr__(CupyGenerator, "zeros", classmethod(zeros))
type.__setattr__(CupyGenerator, "array", classmethod(array))
type.__setattr__(CupyGenerator, "asarray", classmethod(asarray))
type.__setattr__(CupyGenerator, "item", classmethod(item))
type.__setattr__(CupyGenerator, "execute_op", cast("Callable[..., EagerTensorProtocol]", classmethod)(execute_op))

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
