"""Tensor Manipulation operations."""

from ml_switcheroo_compiler.ops.binary import allclose
from ml_switcheroo_compiler.ops.shape import take_along_axis

__all__ = [
    "allclose",
    "take_along_axis",
]
