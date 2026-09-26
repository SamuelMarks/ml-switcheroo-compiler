"""Awkward Array Backend Package for ragged and nested dimension execution."""

from __future__ import annotations

from . import eager, generator, types
from .eager import execute_op
from .generator import AwkwardGenerator
from .types import (
    array,
    asarray,
    broadcast_ragged,
    flatten,
    from_iter,
    from_numpy,
    from_regular,
    get_layout,
    is_ragged,
    item,
    ragged_array,
    ragged_shape,
    record_array,
    to_list,
    to_numpy,
    to_regular,
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
AwkwardGenerator.broadcast_ragged = classmethod(broadcast_ragged)
AwkwardGenerator.is_ragged = classmethod(is_ragged)
AwkwardGenerator.ragged_shape = classmethod(ragged_shape)
AwkwardGenerator.to_regular = classmethod(to_regular)
AwkwardGenerator.from_regular = classmethod(from_regular)
AwkwardGenerator.to_numpy = classmethod(to_numpy)
AwkwardGenerator.from_numpy = classmethod(from_numpy)
AwkwardGenerator.to_list = classmethod(to_list)
AwkwardGenerator.from_iter = classmethod(from_iter)
AwkwardGenerator.get_layout = classmethod(get_layout)

__all__ = [
    "AwkwardGenerator",
    "array",
    "asarray",
    "broadcast_ragged",
    "eager",
    "execute_op",
    "flatten",
    "from_iter",
    "from_numpy",
    "from_regular",
    "generator",
    "get_layout",
    "is_ragged",
    "item",
    "ragged_array",
    "ragged_shape",
    "record_array",
    "to_list",
    "to_numpy",
    "to_regular",
    "types",
    "unflatten",
    "unzip",
    "with_field",
    "zeros",
]
