"""Generate random transformations."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _emit_random_node


def shuffle(key: object, x: object, axis: int = 0) -> object:
    """Shuffles a tensor along a given axis.

    Args:
        key (object): The PRNG key.
        x (object): The input tensor.
        axis (int): The axis to shuffle.

    Returns:
        object: The shuffled tensor.
    """
    return _emit_random_node("RandomShuffle", [key, x], getattr(x, "shape", ()), getattr(x, "dtype", None), {"axis": axis})
