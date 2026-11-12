"""Compiler `multivariate normal` operations and primitives for the `continuous` subsystem."""

from __future__ import annotations

from collections.abc import Sequence

import pydantic

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.random.state import _emit_random_node


class MultivariateNormalOptions(pydantic.BaseModel):
    """Options for multivariate normal."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

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
        key (Tensor | str): Input `key` argument (key).
        mean (Tensor | float): Input `mean` argument (mean).
        cov (Tensor | float): Input `cov` argument (cov).
        options (MultivariateNormalOptions | None): Input `options` argument (options).

    Returns:
        Tensor: Computed `Tensor` output.
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
        inputs,
        out_shape,
        dtype,
        {"method": method},
    )
