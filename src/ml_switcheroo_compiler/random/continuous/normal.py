"""Core abstractions and logic definitions for normal.py."""

from __future__ import annotations

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.random.state import _emit_random_node


def normal(key: object, shape: object = (), dtype: object = None) -> object:
    """Samples standard normal random values from a given key.

    Args:
        key (object): The PRNG key.

        shape (object): The target shape.

        dtype (object): The target data type.



    Returns:
        object: The evaluated output resulting from this operation.

    """
    dtype = dtype or dtypes.DType.Float32
    return _emit_random_node("RandomNormal", [key], shape, dtype)
