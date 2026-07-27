"""Core abstractions and logic definitions for beta.py."""

from __future__ import annotations

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.random.state import _emit_random_node


def beta(key: object, a: object, b: object, shape: object = None, dtype: object = None) -> object:
    """Samples beta random values from a given key."""
    if shape is None:
        shape = ()
    dtype = dtype or dtypes.DType.Float32
    return _emit_random_node("Beta", [key, a, b], shape, dtype)
