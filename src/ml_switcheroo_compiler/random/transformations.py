"""Module transformations.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Generate random transformations."""
from typing import Any

from ml_switcheroo_compiler.random.state import _emit_random_node


def shuffle(key: Any, x: Any, axis: int = 0) -> Any:
    """Shuffles a tensor along a given axis.

    Args:
        key (object): The PRNG key.
        x (object): The input tensor.
        axis (int): The axis to shuffle.

    Returns: Any: The shuffled tensor.
    """
    return _emit_random_node("RandomShuffle", [key, x], getattr(x, "shape", ()), getattr(x, "dtype", None), {"axis": axis})  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
