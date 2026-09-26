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

DPNPGenerator.zeros = classmethod(zeros)
DPNPGenerator.ones = classmethod(ones)
DPNPGenerator.empty = classmethod(empty)
DPNPGenerator.full = classmethod(full)
DPNPGenerator.array = classmethod(array)
DPNPGenerator.asarray = classmethod(asarray)
DPNPGenerator.asnumpy = classmethod(asnumpy)
DPNPGenerator.to_numpy = classmethod(to_numpy)
DPNPGenerator.from_numpy = classmethod(from_numpy)
DPNPGenerator.get_device = classmethod(get_device)
DPNPGenerator.get_usm_type = classmethod(get_usm_type)
DPNPGenerator.is_sycl_array = classmethod(is_sycl_array)
DPNPGenerator.item = classmethod(item)
DPNPGenerator.execute_op = classmethod(execute_op)

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
