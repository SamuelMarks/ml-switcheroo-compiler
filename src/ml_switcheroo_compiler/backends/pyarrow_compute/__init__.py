"""Apache Arrow Compute Backend Package for zero-copy columnar execution."""

from __future__ import annotations

from . import eager, generator, types
from .eager import execute_op
from .generator import PyArrowComputeGenerator
from .types import array, asarray, item, zeros

PyArrowComputeGenerator.zeros = classmethod(zeros)
PyArrowComputeGenerator.array = classmethod(array)
PyArrowComputeGenerator.asarray = classmethod(asarray)
PyArrowComputeGenerator.item = classmethod(item)
PyArrowComputeGenerator.execute_op = classmethod(execute_op)

__all__ = [
    "PyArrowComputeGenerator",
    "array",
    "asarray",
    "eager",
    "execute_op",
    "generator",
    "item",
    "types",
    "zeros",
]
