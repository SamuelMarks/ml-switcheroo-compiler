"""Compiler `uniform` operations and primitives for the `continuous` subsystem."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.random.state import _emit_random_node


def uniform(
    key: Tensor | str,
    shape: Sequence[int] | tuple[int, ...] = (),
    dtype: dtypes.DType | None = None,
    minval: float = 0.0,
    maxval: float = 1.0,
) -> Tensor:
    """Sample uniform random values from a given key.

    Args:
        key (Tensor | str): Input `key` argument (key).
        shape (Sequence[int] | tuple[int, ...]): Input `shape` argument (shape).
        dtype (dtypes.DType | None): Input `dtype` argument (dtype).
        minval (float): Input `minval` argument (minval).
        maxval (float): Input `maxval` argument (maxval).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    dtype = dtype or dtypes.DType.Float32
    return _emit_random_node(
        "RandomUniform",
        [key],
        shape,
        dtype,
        {"minval": minval, "maxval": maxval},
    )
