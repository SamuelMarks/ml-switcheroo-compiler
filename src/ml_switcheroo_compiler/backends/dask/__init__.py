# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module __init__.py."""

"""Dask Code Generator Package."""

from .eager import execute_op
from .generator import DaskGenerator
from .types import array, asarray, item, zeros

DaskGenerator.zeros = classmethod(zeros)
DaskGenerator.array = classmethod(array)
DaskGenerator.asarray = classmethod(asarray)
DaskGenerator.item = classmethod(item)
DaskGenerator.execute_op = classmethod(execute_op)
