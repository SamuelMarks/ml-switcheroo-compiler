"""Module docstring."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.random.state import _emit_random_node


def multivariate_normal(
    key: object,
    mean: object,
    cov: object,
    shape: object = None,
    dtype: object = None,
    method: str = "cholesky",
) -> object:
    """Samples from a multivariate normal distribution.

    # pragma: no cover
    Args:
    # pragma: no cover
        key (object): The PRNG key.
    # pragma: no cover
        mean (object): Mean vector of the distribution.
    # pragma: no cover
        cov (object): Covariance matrix of the distribution.
    # pragma: no cover
        shape (object): Target shape.
    # pragma: no cover
        dtype (object): Target data type.
    # pragma: no cover
        method (str): Matrix decomposition method ('svd', 'eigh', 'cholesky').
    # pragma: no cover

    # pragma: no cover
    Returns:
    # pragma: no cover
        object: Sampled tensor.
    # pragma: no cover
    """
    dtype = dtype or dtypes.DType.Float32
    if config.eager_mode:
        mean_data = getattr(mean, "data", mean)
        cov_data = getattr(cov, "data", cov)
        seed_val = int(key.data[1]) if isinstance(key, Tensor) else 0
        rng = np.random.default_rng(seed_val)
        batch_shape = shape if shape is not None else ()
        res = rng.multivariate_normal(mean_data, cov_data, size=batch_shape, method=method)
        return Tensor(res, TensorConfig(res.shape, dtype, config.default_device))
    out_shape = shape if shape is not None else ()
    inputs = [key]
    if isinstance(mean, Tensor):
        inputs.append(mean)
    if isinstance(cov, Tensor):
        inputs.append(cov)
    return _emit_random_node("MultivariateNormal", inputs, out_shape, dtype, {"method": method})
