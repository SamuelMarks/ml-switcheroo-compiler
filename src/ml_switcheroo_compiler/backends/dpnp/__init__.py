"""Data Parallel NumPy (dpnp) Backend Package for Intel SYCL hardware."""

from __future__ import annotations

from . import eager, generator, types
from .eager import execute_op
from .generator import DPNPGenerator
from .types import array, asarray, item, zeros

DPNPGenerator.zeros = classmethod(zeros)
DPNPGenerator.array = classmethod(array)
DPNPGenerator.asarray = classmethod(asarray)
DPNPGenerator.item = classmethod(item)
DPNPGenerator.execute_op = classmethod(execute_op)

__all__ = [
    "DPNPGenerator",
    "array",
    "asarray",
    "eager",
    "execute_op",
    "generator",
    "item",
    "types",
    "zeros",
]
