"""Module docstring."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.random.state import _emit_random_node, _get_numpy_rng


def dirichlet(key: object, alpha: object, shape: object = None, dtype: object = None) -> object:
    """Samples dirichlet random values from a given key."""
    if shape is None:
        shape = ()
    dtype = dtype or dtypes.DType.Float32
    if config.eager_mode:
        np_dtype = np.dtype(dtype.value)
        alpha_val = getattr(alpha, "data", alpha)

        rng = _get_numpy_rng(key)
        res = np.asarray(rng.dirichlet(alpha_val, size=shape)).astype(np_dtype)
        return Tensor(res, TensorConfig(getattr(res, "shape", ()), dtype, config.default_device))
    return _emit_random_node("Dirichlet", [key, alpha], shape, dtype)
