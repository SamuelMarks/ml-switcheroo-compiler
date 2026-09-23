"""Awkward Array Backend Package for ragged and nested dimension execution."""

from __future__ import annotations

from . import eager, generator, types
from .eager import execute_op
from .generator import AwkwardGenerator
from .types import (
    array,
    asarray,
    flatten,
    item,
    ragged_array,
    record_array,
    unflatten,
    unzip,
    with_field,
    zeros,
)

AwkwardGenerator.zeros = classmethod(zeros)
AwkwardGenerator.array = classmethod(array)
AwkwardGenerator.asarray = classmethod(asarray)
AwkwardGenerator.ragged_array = classmethod(ragged_array)
AwkwardGenerator.record_array = classmethod(record_array)
AwkwardGenerator.flatten = classmethod(flatten)
AwkwardGenerator.unflatten = classmethod(unflatten)
AwkwardGenerator.with_field = classmethod(with_field)
AwkwardGenerator.unzip = classmethod(unzip)
AwkwardGenerator.item = classmethod(item)
AwkwardGenerator.execute_op = classmethod(execute_op)

__all__ = [
    "AwkwardGenerator",
    "array",
    "asarray",
    "eager",
    "execute_op",
    "flatten",
    "generator",
    "item",
    "ragged_array",
    "record_array",
    "types",
    "unflatten",
    "unzip",
    "with_field",
    "zeros",
]
