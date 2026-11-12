"""Compiler `truncated normal` operations and primitives for the `continuous` subsystem."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.random.state import _emit_random_node


def truncated_normal(
    key: Tensor | str,
    lower: float,
    upper: float,
    shape: Sequence[int] | tuple[int, ...] = (),
    dtype: dtypes.DType | None = None,
) -> Tensor:
    """Return an initializer that generates arrays from a truncated normal distribution.

    Args:
        key (Tensor | str): Input `key` argument (key).
        lower (float): Input `lower` argument (lower).
        upper (float): Input `upper` argument (upper).
        shape (Sequence[int] | tuple[int, ...]): Input `shape` argument (shape).
        dtype (dtypes.DType | None): Input `dtype` argument (dtype).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    dtype = dtype or dtypes.DType.Float32
    return _emit_random_node(
        "RandomTruncatedNormal",
        [key],
        shape,
        dtype,
        {"lower": lower, "upper": upper},
    )
