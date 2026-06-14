"""Mlx Code Generator Package."""

from .eager import execute_op
from .generator import MLXCodeGenerator
from .types import array, asarray, item, zeros

MLXCodeGenerator.zeros = classmethod(zeros)
MLXCodeGenerator.array = classmethod(array)
MLXCodeGenerator.asarray = classmethod(asarray)
MLXCodeGenerator.item = classmethod(item)
MLXCodeGenerator.execute_op = classmethod(execute_op)

__all__ = ["MLXCodeGenerator"]
