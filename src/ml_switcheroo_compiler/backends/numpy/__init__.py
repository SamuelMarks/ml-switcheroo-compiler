"""Numpy Code Generator Package."""

from .eager import execute_op
from .generator import NumpyGenerator
from .types import array, asarray, item, zeros

NumpyGenerator.execute_op = classmethod(execute_op)
NumpyGenerator.zeros = classmethod(zeros)
NumpyGenerator.array = classmethod(array)
NumpyGenerator.asarray = classmethod(asarray)
NumpyGenerator.item = classmethod(item)

__all__ = ["NumpyGenerator"]
