"""Module double_sided_maxwell.py."""

from __future__ import annotations

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.random.state import _dispatch_random


def double_sided_maxwell(*args: Tensor | float | int | str, **kwargs: Tensor | float | int | str) -> Tensor:
    """Evaluate double_sided_maxwell operation.

    Args:
        *args (Tensor | float | int | str): Positional args.
        **kwargs (Tensor | float | int | str): Keyword args.

    Returns:
        Tensor: Result.
    """
    return _dispatch_random("double_sided_maxwell", *args, **kwargs)
