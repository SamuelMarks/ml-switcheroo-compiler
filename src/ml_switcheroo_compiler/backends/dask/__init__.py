# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
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
