"""Bohrium Backend Package for multi-core and OpenCL lazy evaluation."""

from __future__ import annotations

from . import eager, generator, types
from .eager import execute_op
from .generator import BohriumGenerator
from .types import array, asarray, item, zeros

BohriumGenerator.zeros = classmethod(zeros)
BohriumGenerator.array = classmethod(array)
BohriumGenerator.asarray = classmethod(asarray)
BohriumGenerator.item = classmethod(item)
BohriumGenerator.execute_op = classmethod(execute_op)

__all__ = [
    "BohriumGenerator",
    "array",
    "asarray",
    "eager",
    "execute_op",
    "generator",
    "item",
    "types",
    "zeros",
]
