"""Core abstractions and logic definitions for uniform.py."""

from __future__ import annotations

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.random.state import _emit_random_node


def uniform(key: object, shape: object = (), dtype: object = None, minval: object = 0.0, maxval: object = 1.0) -> object:
    """Sample uniform random values from a given key.

    Args:
        key (object): The key parameter.
        shape (object): The shape parameter.
        dtype (object): The dtype parameter.
        minval (object): The minval parameter.
        maxval (object): The maxval parameter.

    Returns:
        object: Result.
    """
    dtype = dtype or dtypes.DType.Float32
    return _emit_random_node("RandomUniform", [key], shape, dtype, {"minval": minval, "maxval": maxval})
