# ruff: noqa: E501
"""Dask Code Generator Package."""

from .eager import execute_op
from .generator import DaskGenerator
from .types import array, asarray, item, zeros

DaskGenerator.zeros = classmethod(zeros)
DaskGenerator.array = classmethod(array)
DaskGenerator.asarray = classmethod(asarray)
DaskGenerator.item = classmethod(item)
DaskGenerator.execute_op = classmethod(execute_op)
