"""Core abstractions and logic definitions for multivariate_normal.py."""

from __future__ import annotations

from dataclasses import dataclass

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.random.state import _emit_random_node


@dataclass
class MultivariateNormalOptions:
    """Options for multivariate normal."""

    shape: object | None = None
    dtype: object | None = None
    method: str = "cholesky"


def multivariate_normal(key: object, mean: object, cov: object, options: MultivariateNormalOptions | None = None) -> object:
    """Sample from a multivariate normal distribution.

    Args:
        key (object): The key parameter.
        mean (object): The mean parameter.
        cov (object): The cov parameter.
        options (object): The options parameter.

    Returns:
        object: Result.
    """
    options = options or MultivariateNormalOptions()
    shape = options.shape
    dtype = options.dtype
    method = options.method

    dtype = dtype or dtypes.DType.Float32
    out_shape = shape if shape is not None else ()
    inputs = [key]
    if isinstance(mean, Tensor):
        inputs.append(mean)
    if isinstance(cov, Tensor):
        inputs.append(cov)
    return _emit_random_node("MultivariateNormal", inputs, out_shape, dtype, {"method": method})
