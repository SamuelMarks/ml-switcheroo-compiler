"""Core abstractions and logic definitions for beta.py."""

from __future__ import annotations

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.random.state import _emit_random_node


def beta(key: object, a: object, b: object, shape: object = None, dtype: object = None) -> object:
    """Sample beta random values from a given key.

    Args:
        key (object): The key parameter.
        a (object): The a parameter.
        b (object): The b parameter.
        shape (object): The shape parameter.
        dtype (object): The dtype parameter.

    Returns:
        object: Result.
    """
    if shape is None:
        shape = ()
    dtype = dtype or dtypes.DType.Float32
    return _emit_random_node("Beta", [key, a, b], shape, dtype)
