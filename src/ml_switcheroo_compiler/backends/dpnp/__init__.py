"""Data Parallel NumPy (dpnp) Backend Package for Intel SYCL hardware."""

from __future__ import annotations

from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol

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

type.__setattr__(DPNPGenerator, "zeros", staticmethod(zeros))
type.__setattr__(DPNPGenerator, "ones", staticmethod(ones))
type.__setattr__(DPNPGenerator, "empty", staticmethod(empty))
type.__setattr__(DPNPGenerator, "full", staticmethod(full))
type.__setattr__(DPNPGenerator, "array", staticmethod(array))
type.__setattr__(DPNPGenerator, "asarray", staticmethod(asarray))
type.__setattr__(DPNPGenerator, "asnumpy", staticmethod(asnumpy))
type.__setattr__(DPNPGenerator, "to_numpy", staticmethod(to_numpy))
type.__setattr__(DPNPGenerator, "from_numpy", staticmethod(from_numpy))
type.__setattr__(DPNPGenerator, "get_device", staticmethod(get_device))
type.__setattr__(DPNPGenerator, "get_usm_type", staticmethod(get_usm_type))
type.__setattr__(DPNPGenerator, "is_sycl_array", staticmethod(is_sycl_array))
type.__setattr__(DPNPGenerator, "item", staticmethod(item))
type.__setattr__(DPNPGenerator, "execute_op", cast("EagerCallableProtocol", staticmethod)(execute_op))

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
