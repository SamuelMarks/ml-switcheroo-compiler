"""Module generalized_normal.py."""

from __future__ import annotations

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.random.state import _dispatch_random


def generalized_normal(*args: Tensor | float | int | str, **kwargs: Tensor | float | int | str) -> Tensor:
    """Evaluate generalized_normal operation.

    Args:
        *args (Tensor | float | int | str): Positional args.
        **kwargs (Tensor | float | int | str): Keyword args.

    Returns:
        Tensor: Result.
    """
    return _dispatch_random("generalized_normal", *args, **kwargs)
