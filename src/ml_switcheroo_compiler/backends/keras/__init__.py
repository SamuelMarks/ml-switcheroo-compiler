"""Keras Code Generator Package."""

from .eager import execute_op
from .generator import KerasCodeGenerator
from .types import array, asarray, item, zeros

KerasCodeGenerator.zeros = classmethod(zeros)
KerasCodeGenerator.array = classmethod(array)
KerasCodeGenerator.asarray = classmethod(asarray)
KerasCodeGenerator.item = classmethod(item)
KerasCodeGenerator.execute_op = classmethod(execute_op)

__all__ = ["KerasCodeGenerator"]
