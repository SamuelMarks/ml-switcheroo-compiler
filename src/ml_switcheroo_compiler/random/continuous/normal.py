"""Module docstring."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.random.state import _emit_random_node


def normal(key: object, shape: object = (), dtype: object = None) -> object:
    """Samples standard normal random values from a given key.

    # pragma: no cover
    Args:
    # pragma: no cover
        key (object): The PRNG key.
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
        np_dtype = np.dtype(dtype.value)
        key_data = getattr(key, "data", key)
        seed = [int(x) for x in np.asarray(key_data).ravel()] if np.ndim(key_data) > 0 else None
        rng = np.random.default_rng(seed)
        return Tensor(
            rng.normal(size=shape).astype(np_dtype),
            TensorConfig(shape if shape is not None else (), dtype, config.default_device),
        )
    return _emit_random_node("RandomNormal", [key], shape, dtype)
