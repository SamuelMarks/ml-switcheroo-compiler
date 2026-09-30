"""Module multivariate_normal.py."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.random.state import _emit_random_node


@dataclass
class MultivariateNormalOptions:
    """Options for multivariate normal."""

    shape: Sequence[int] | tuple[int, ...] | None = None
    dtype: dtypes.DType | None = None
    method: str = "cholesky"


def multivariate_normal(
    key: Tensor | str,
    mean: Tensor | float,
    cov: Tensor | float,
    options: MultivariateNormalOptions | None = None,
) -> Tensor:
    """Sample from a multivariate normal distribution.

    Args:
        key (Tensor | str): The key parameter.
        mean (Tensor | float): The mean parameter.
        cov (Tensor | float): The cov parameter.
        options (MultivariateNormalOptions | None): The options parameter.

    Returns:
        Tensor: Result.
    """
    options = options or MultivariateNormalOptions()
    shape = options.shape
    dtype = options.dtype
    method = options.method

    dtype = dtype or dtypes.DType.Float32
    out_shape = tuple(shape) if shape is not None else ()
    inputs: list[Tensor | str] = [key]
    if isinstance(mean, Tensor):
        inputs.append(mean)
    if isinstance(cov, Tensor):
        inputs.append(cov)
    return _emit_random_node(
        "MultivariateNormal",
        inputs,  # type: ignore[arg-type]
        out_shape,
        dtype,
        {"method": method},
    )
