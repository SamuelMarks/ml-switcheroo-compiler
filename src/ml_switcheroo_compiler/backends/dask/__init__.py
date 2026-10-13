# ruff: noqa: E402, E501, F403
"""Dask Code Generator Package."""

import importlib.util
import sys

try:
    _has_pkg = importlib.util.find_spec("dask") is not None
except ValueError:
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'dask' backend requires the 'dask' library to be installed.")

from .eager import execute_op
from .generator import DaskGenerator
from .types import array, asarray, item, zeros

DaskGenerator.zeros = classmethod(zeros)
DaskGenerator.array = classmethod(array)
DaskGenerator.asarray = classmethod(asarray)
DaskGenerator.item = classmethod(item)
DaskGenerator.execute_op = classmethod(execute_op)
