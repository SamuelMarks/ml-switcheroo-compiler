"""Module truncated_normal.py."""

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
        key (Tensor | str): The key parameter.
        lower (float): The lower parameter.
        upper (float): The upper parameter.
        shape (Sequence[int] | tuple[int, ...]): The shape parameter.
        dtype (dtypes.DType | None): The dtype parameter.

    Returns:
        Tensor: Result.
    """
    dtype = dtype or dtypes.DType.Float32
    return _emit_random_node(
        "RandomTruncatedNormal",
        [key],  # type: ignore[list-item]
        shape,
        dtype,
        {"lower": lower, "upper": upper},
    )
