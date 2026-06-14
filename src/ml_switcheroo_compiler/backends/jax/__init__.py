"""Jax Code Generator Package."""

from .eager import execute_op
from .generator import JAXCodeGenerator
from .types import array, asarray, item, zeros

JAXCodeGenerator.zeros = classmethod(zeros)
JAXCodeGenerator.array = classmethod(array)
JAXCodeGenerator.asarray = classmethod(asarray)
JAXCodeGenerator.item = classmethod(item)
JAXCodeGenerator.execute_op = classmethod(execute_op)

__all__ = ["JAXCodeGenerator"]
