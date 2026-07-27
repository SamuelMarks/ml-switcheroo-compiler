"""Core abstractions and logic definitions for gamma.py."""

from __future__ import annotations

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.random.state import _emit_random_node


def gamma(key: object, a: object, shape: object = (), dtype: object = None) -> object:
    """Samples gamma random values from a given key."""
    dtype = dtype or dtypes.DType.Float32
    return _emit_random_node("Gamma", [key, a], shape, dtype)
