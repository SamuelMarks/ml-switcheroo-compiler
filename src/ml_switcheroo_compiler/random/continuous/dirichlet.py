"""Core abstractions and logic definitions for dirichlet.py."""

from __future__ import annotations

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.random.state import _emit_random_node


def dirichlet(key: object, alpha: object, shape: object = None, dtype: object = None) -> object:
    """Sample dirichlet random values from a given key.

    Args:
        key (object): The key parameter.
        alpha (object): The alpha parameter.
        shape (object): The shape parameter.
        dtype (object): The dtype parameter.

    Returns:
        object: Result.
    """
    if shape is None:
        shape = ()
    dtype = dtype or dtypes.DType.Float32
    return _emit_random_node("Dirichlet", [key, alpha], shape, dtype)
