"""Tensor Manipulation operations."""

from ml_switcheroo_compiler.ops.shape import take_along_axis
from ml_switcheroo_compiler.ops.binary import allclose

__all__ = [
    "allclose",
    "take_along_axis",
]
