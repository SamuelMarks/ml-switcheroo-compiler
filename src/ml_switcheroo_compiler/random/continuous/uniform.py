"""Module uniform.py."""

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
        key (Tensor | str): The key parameter.
        shape (Sequence[int] | tuple[int, ...]): The shape parameter.
        dtype (dtypes.DType | None): The dtype parameter.
        minval (float): The minval parameter.
        maxval (float): The maxval parameter.

    Returns:
        Tensor: Result.
    """
    dtype = dtype or dtypes.DType.Float32
    return _emit_random_node(
        "RandomUniform",
        [key],  # type: ignore[list-item]
        shape,
        dtype,
        {"minval": minval, "maxval": maxval},
    )
