"""Module transformations.py."""

from __future__ import annotations

from ml_switcheroo_compiler.core.tensor import Tensor
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
    return _emit_random_node(
        "RandomShuffle",
        [key, x],  # type: ignore[list-item]
        getattr(x, "shape", ()),
        getattr(x, "dtype", None),
        {"axis": axis},
    )
