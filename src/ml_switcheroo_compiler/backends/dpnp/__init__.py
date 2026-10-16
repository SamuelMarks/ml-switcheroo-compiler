"""Data Parallel NumPy (dpnp) Backend Package for Intel SYCL hardware."""

from __future__ import annotations

from typing import Callable, cast

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol

from . import eager, generator, types
from .eager import execute_op
from .generator import DPNPGenerator
from .types import (
    array,
    asarray,
    asnumpy,
    empty,
    from_numpy,
    full,
    get_device,
    get_usm_type,
    is_sycl_array,
    item,
    ones,
    to_numpy,
    zeros,
)

type.__setattr__(DPNPGenerator, "zeros", classmethod(zeros))
type.__setattr__(DPNPGenerator, "ones", classmethod(ones))
type.__setattr__(DPNPGenerator, "empty", classmethod(empty))
type.__setattr__(DPNPGenerator, "full", classmethod(full))
type.__setattr__(DPNPGenerator, "array", classmethod(array))
type.__setattr__(DPNPGenerator, "asarray", classmethod(asarray))
type.__setattr__(DPNPGenerator, "asnumpy", classmethod(asnumpy))
type.__setattr__(DPNPGenerator, "to_numpy", classmethod(to_numpy))
type.__setattr__(DPNPGenerator, "from_numpy", classmethod(from_numpy))
type.__setattr__(DPNPGenerator, "get_device", classmethod(get_device))
type.__setattr__(DPNPGenerator, "get_usm_type", classmethod(get_usm_type))
type.__setattr__(DPNPGenerator, "is_sycl_array", classmethod(is_sycl_array))
type.__setattr__(DPNPGenerator, "item", classmethod(item))
type.__setattr__(DPNPGenerator, "execute_op", cast("Callable[..., EagerTensorProtocol]", classmethod)(execute_op))

__all__ = [
    "DPNPGenerator",
    "array",
    "asarray",
    "asnumpy",
    "eager",
    "empty",
    "execute_op",
    "from_numpy",
    "full",
    "generator",
    "get_device",
    "get_usm_type",
    "is_sycl_array",
    "item",
    "ones",
    "to_numpy",
    "types",
    "zeros",
]
