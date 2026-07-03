"""Module docstring."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.random.state import _emit_random_node


def _eager_truncated_normal(rng: object, shape: object, dtype: object, lower: object, upper: object) -> object:
    """Function docstring.

    # pragma: no cover
    Args:
    # pragma: no cover
        rng: Arg.
    # pragma: no cover
        shape: Arg.
    # pragma: no cover
        dtype: Arg.
    # pragma: no cover
        lower: Arg.
    # pragma: no cover
        upper: Arg.
    # pragma: no cover
    """
    np_dtype = np.dtype(dtype.value)
    arr = rng.normal(size=shape)
    low = getattr(lower, "data", lower)
    up = getattr(upper, "data", upper)
    low = np.broadcast_to(low, shape) if np.ndim(low) > 0 else low
    up = np.broadcast_to(up, shape) if np.ndim(up) > 0 else up
    is_invalid = low >= up
    safe_up = np.where(is_invalid, low + 1.0, up)
    mask = (arr < low) | (arr > safe_up)
    while np.any(mask):
        arr[mask] = rng.normal(size=np.count_nonzero(mask))
        mask = (arr < low) | (arr > safe_up)
    arr = np.where(is_invalid, low, arr)
    return Tensor(
        arr.astype(np_dtype),
        TensorConfig(shape if shape is not None else (), dtype, config.default_device),
    )


def truncated_normal(key: object, lower: object, upper: object, shape: object = (), dtype: object = None) -> object:
    """Returns an initializer that generates arrays from a truncated normal distribution.

    # pragma: no cover
    Args:
    # pragma: no cover
        key (object): The PRNG key.
    # pragma: no cover
        lower (object): The lower parameter for the operation.
    # pragma: no cover
        upper (object): The upper parameter for the operation.
    # pragma: no cover
        shape (object): The target shape.
    # pragma: no cover
        dtype (object): The target data type.
    # pragma: no cover

    # pragma: no cover
    Returns:
    # pragma: no cover
        object: The evaluated output resulting from this operation.
    # pragma: no cover
    """
    dtype = dtype or dtypes.DType.Float32
    if config.eager_mode:
        key_data = getattr(key, "data", key)
        seed = [int(x) for x in np.asarray(key_data).ravel()] if np.ndim(key_data) > 0 else None
        rng = np.random.default_rng(seed)
        return _eager_truncated_normal(rng, shape, dtype, lower, upper)
    return _emit_random_node("RandomTruncatedNormal", [key], shape, dtype, {"lower": lower, "upper": upper})
