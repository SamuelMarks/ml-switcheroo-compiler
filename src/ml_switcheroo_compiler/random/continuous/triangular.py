"""Module triangular.py."""

from __future__ import annotations

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.random.state import _dispatch_random


def triangular(*args: Tensor | float | int | str, **kwargs: Tensor | float | int | str) -> Tensor:
    """Evaluate triangular operation.

    Args:
        *args (Tensor | float | int | str): Positional args.
        **kwargs (Tensor | float | int | str): Keyword args.

    Returns:
        Tensor: Result.
    """
    return _dispatch_random("triangular", *args, **kwargs)
