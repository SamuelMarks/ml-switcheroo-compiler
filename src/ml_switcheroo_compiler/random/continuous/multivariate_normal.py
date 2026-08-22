"""Module multivariate_normal.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for multivariate_normal.py."""
from dataclasses import dataclass
from typing import Any

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.random.state import _emit_random_node


@dataclass
class MultivariateNormalOptions:
    """Options for multivariate normal."""

    shape: Any | None = None
    dtype: Any | None = None
    method: str = "cholesky"


def multivariate_normal(key: Any, mean: Any, cov: Any, options: MultivariateNormalOptions | None = None) -> Any:
    """Sample from a multivariate normal distribution.

    Args:
        key (object): The key parameter.
        mean (object): The mean parameter.
        cov (object): The cov parameter.
        options (object): The options parameter.

    Returns:
            tuple[int, ...]: Result.
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
