"""Module docstring."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.random.state import _emit_random_node, _get_numpy_rng


def gamma(key: object, a: object, shape: object = (), dtype: object = None) -> object:
    """Samples gamma random values from a given key."""
    dtype = dtype or dtypes.DType.Float32
    if config.eager_mode:
        np_dtype = np.dtype(dtype.value)
        a_val = getattr(a, "data", a)

        rng = _get_numpy_rng(key)
        res = np.asarray(rng.gamma(a_val, size=shape)).astype(np_dtype)
        return Tensor(res, TensorConfig(shape if shape is not None else (), dtype, config.default_device))
    return _emit_random_node("Gamma", [key, a], shape, dtype)
