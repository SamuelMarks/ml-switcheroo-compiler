"""Module normal.py."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.random.state import _emit_random_node


def normal(
    key: Tensor | str,
    shape: Sequence[int] | tuple[int, ...] = (),
    dtype: dtypes.DType | None = None,
) -> Tensor:
    """Sample standard normal random values from a given key.

    Args:
        key (Tensor | str): The key parameter.
        shape (Sequence[int] | tuple[int, ...]): The shape parameter.
        dtype (dtypes.DType | None): The dtype parameter.

    Returns:
        Tensor: Result.
    """
    dtype = dtype or dtypes.DType.Float32
    return _emit_random_node("RandomNormal", [key], shape, dtype)  # type: ignore[list-item]
