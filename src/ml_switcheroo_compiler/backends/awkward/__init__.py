"""Awkward Array Backend Package for ragged and nested dimension structures."""

from __future__ import annotations

from . import eager, generator, types
from .eager import execute_op
from .generator import AwkwardGenerator
from .types import array, asarray, item, zeros

AwkwardGenerator.zeros = classmethod(zeros)
AwkwardGenerator.array = classmethod(array)
AwkwardGenerator.asarray = classmethod(asarray)
AwkwardGenerator.item = classmethod(item)
AwkwardGenerator.execute_op = classmethod(execute_op)

__all__ = [
    "AwkwardGenerator",
    "array",
    "asarray",
    "eager",
    "execute_op",
    "generator",
    "item",
    "types",
    "zeros",
]
