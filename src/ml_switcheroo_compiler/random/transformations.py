"""Module transformations.py."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.eager_registry import (
    ShapeProtocol,
    StructuralOperandProtocol,
    as_shape,
)
from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.random.distributions_discrete import _RandomTensorProtocol
from ml_switcheroo_compiler.random.state import _emit_random_node


def shuffle(key: Tensor | str, x: Tensor, axis: int = 0) -> Tensor:
    """Shuffles a tensor along a given axis.

    Args:
        key (Tensor | str): The PRNG key.
        x (Tensor): The input tensor.
        axis (int): The axis to shuffle.

    Returns:
        Tensor: The shuffled tensor.
    """
    raw_x: StructuralOperandProtocol = x
    shape: tuple[int, ...] = as_shape(raw_x.shape) if isinstance(raw_x, (Tensor, ShapeProtocol, _RandomTensorProtocol)) else ()
    dtype: dtypes.DType | str | None = raw_x.dtype if isinstance(raw_x, (Tensor, _RandomTensorProtocol)) else None
    return _emit_random_node(
        "RandomShuffle",
        [key, x],
        shape,
        dtype,
        {"axis": axis},
    )
