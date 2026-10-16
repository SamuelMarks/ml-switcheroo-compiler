"""Awkward Array Backend Package for ragged and nested dimension execution."""

from __future__ import annotations

from typing import Callable, cast

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol

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

type.__setattr__(AwkwardGenerator, "zeros", classmethod(zeros))
type.__setattr__(AwkwardGenerator, "array", classmethod(array))
type.__setattr__(AwkwardGenerator, "asarray", classmethod(asarray))
type.__setattr__(AwkwardGenerator, "ragged_array", classmethod(ragged_array))
type.__setattr__(AwkwardGenerator, "record_array", classmethod(record_array))
type.__setattr__(AwkwardGenerator, "flatten", classmethod(flatten))
type.__setattr__(AwkwardGenerator, "unflatten", classmethod(unflatten))
type.__setattr__(AwkwardGenerator, "with_field", classmethod(with_field))
type.__setattr__(AwkwardGenerator, "unzip", classmethod(unzip))
type.__setattr__(AwkwardGenerator, "item", classmethod(item))
type.__setattr__(AwkwardGenerator, "execute_op", cast("Callable[..., EagerTensorProtocol]", classmethod)(execute_op))
type.__setattr__(AwkwardGenerator, "broadcast_ragged", classmethod(broadcast_ragged))
type.__setattr__(AwkwardGenerator, "is_ragged", classmethod(is_ragged))
type.__setattr__(AwkwardGenerator, "ragged_shape", classmethod(ragged_shape))
type.__setattr__(AwkwardGenerator, "to_regular", classmethod(to_regular))
type.__setattr__(AwkwardGenerator, "from_regular", classmethod(from_regular))
type.__setattr__(AwkwardGenerator, "to_numpy", classmethod(to_numpy))
type.__setattr__(AwkwardGenerator, "from_numpy", classmethod(from_numpy))
type.__setattr__(AwkwardGenerator, "to_list", classmethod(to_list))
type.__setattr__(AwkwardGenerator, "from_iter", classmethod(from_iter))
type.__setattr__(AwkwardGenerator, "get_layout", classmethod(get_layout))

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
