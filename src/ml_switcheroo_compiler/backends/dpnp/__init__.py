"""Data Parallel NumPy (dpnp) Backend Package for Intel SYCL hardware."""

from __future__ import annotations

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
