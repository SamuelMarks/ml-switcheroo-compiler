"""Module dirichlet.py."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.random.state import _emit_random_node


def dirichlet(
    key: Tensor | str,
    alpha: Tensor | float,
    shape: Sequence[int] | tuple[int, ...] | int | None = None,
    dtype: dtypes.DType | None = None,
) -> Tensor:
    """Sample dirichlet random values from a given key.

    Args:
        key (Tensor | str): The key parameter.
        alpha (Tensor | float): The alpha parameter.
        shape (Sequence[int] | tuple[int, ...] | int | None): The shape parameter.
        dtype (dtypes.DType | None): The dtype parameter.

    Returns:
        Tensor: Result.
    """
    if shape is None:
        shape = ()
    dtype = dtype or dtypes.DType.Float32
    return _emit_random_node("Dirichlet", [key, alpha], shape, dtype)  # type: ignore[list-item]
