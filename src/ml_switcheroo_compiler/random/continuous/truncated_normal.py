"""Core abstractions and logic definitions for truncated_normal.py."""

from __future__ import annotations

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.random.state import _emit_random_node


def truncated_normal(key: object, lower: object, upper: object, shape: object = (), dtype: object = None) -> object:
    """Return an initializer that generates arrays from a truncated normal distribution.

    Args:
        key (object): The key parameter.
        lower (object): The lower parameter.
        upper (object): The upper parameter.
        shape (object): The shape parameter.
        dtype (object): The dtype parameter.

    Returns:
        object: Result.
    """
    dtype = dtype or dtypes.DType.Float32
    return _emit_random_node("RandomTruncatedNormal", [key], shape, dtype, {"lower": lower, "upper": upper})
