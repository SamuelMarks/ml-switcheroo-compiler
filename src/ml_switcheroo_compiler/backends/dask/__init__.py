"""Dask Code Generator Package."""

import importlib.util
import sys
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol

try:
    _has_pkg = importlib.util.find_spec("dask") is not None
except ValueError:
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'dask' backend requires the 'dask' library to be installed.")

from .eager import execute_op
from .generator import DaskGenerator
from .types import array, asarray, item, zeros

type.__setattr__(DaskGenerator, "zeros", classmethod(zeros))
type.__setattr__(DaskGenerator, "array", classmethod(array))
type.__setattr__(DaskGenerator, "asarray", classmethod(asarray))
type.__setattr__(DaskGenerator, "item", classmethod(item))
type.__setattr__(DaskGenerator, "execute_op", cast("EagerCallableProtocol", classmethod)(execute_op))
