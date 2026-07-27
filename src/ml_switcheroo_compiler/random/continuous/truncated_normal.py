"""Core abstractions and logic definitions for truncated_normal.py."""

from __future__ import annotations

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.random.state import _emit_random_node


def truncated_normal(key: object, lower: object, upper: object, shape: object = (), dtype: object = None) -> object:
    """Returns an initializer that generates arrays from a truncated normal distribution.

    Args:
        key (object): The PRNG key.

        lower (object): The lower parameter for the operation.

        upper (object): The upper parameter for the operation.

        shape (object): The target shape.

        dtype (object): The target data type.



    Returns:
        object: The evaluated output resulting from this operation.

    """
    dtype = dtype or dtypes.DType.Float32
    return _emit_random_node("RandomTruncatedNormal", [key], shape, dtype, {"lower": lower, "upper": upper})
