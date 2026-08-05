"""Core abstractions and logic definitions for gamma.py."""

from __future__ import annotations

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.random.state import _emit_random_node


def gamma(key: object, a: object, shape: object = (), dtype: object = None) -> object:
    """Sample gamma random values from a given key.

    Args:
        key (object): The key parameter.
        a (object): The a parameter.
        shape (object): The shape parameter.
        dtype (object): The dtype parameter.

    Returns:
        object: Result.
    """
    dtype = dtype or dtypes.DType.Float32
    return _emit_random_node("Gamma", [key, a], shape, dtype)
