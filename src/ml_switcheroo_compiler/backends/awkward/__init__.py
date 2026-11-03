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
