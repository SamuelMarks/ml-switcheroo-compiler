"""Awkward Array Backend Package for ragged and nested dimension execution."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol

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

type.__setattr__(AwkwardGenerator, "zeros", staticmethod(zeros))
type.__setattr__(AwkwardGenerator, "array", staticmethod(array))
type.__setattr__(AwkwardGenerator, "asarray", staticmethod(asarray))
type.__setattr__(AwkwardGenerator, "ragged_array", staticmethod(ragged_array))
type.__setattr__(AwkwardGenerator, "record_array", staticmethod(record_array))
type.__setattr__(AwkwardGenerator, "flatten", staticmethod(flatten))
type.__setattr__(AwkwardGenerator, "unflatten", staticmethod(unflatten))
type.__setattr__(AwkwardGenerator, "with_field", staticmethod(with_field))
type.__setattr__(AwkwardGenerator, "unzip", staticmethod(unzip))
type.__setattr__(AwkwardGenerator, "item", staticmethod(item))
type.__setattr__(AwkwardGenerator, "execute_op", typing.cast("EagerCallableProtocol", staticmethod)(execute_op))
type.__setattr__(AwkwardGenerator, "broadcast_ragged", staticmethod(broadcast_ragged))
type.__setattr__(AwkwardGenerator, "is_ragged", staticmethod(is_ragged))
type.__setattr__(AwkwardGenerator, "ragged_shape", staticmethod(ragged_shape))
type.__setattr__(AwkwardGenerator, "to_regular", staticmethod(to_regular))
type.__setattr__(AwkwardGenerator, "from_regular", staticmethod(from_regular))
type.__setattr__(AwkwardGenerator, "to_numpy", staticmethod(to_numpy))
type.__setattr__(AwkwardGenerator, "from_numpy", staticmethod(from_numpy))
type.__setattr__(AwkwardGenerator, "to_list", staticmethod(to_list))
type.__setattr__(AwkwardGenerator, "from_iter", staticmethod(from_iter))
type.__setattr__(AwkwardGenerator, "get_layout", staticmethod(get_layout))

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
