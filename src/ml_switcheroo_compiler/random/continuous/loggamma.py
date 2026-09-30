"""Module loggamma.py."""

from __future__ import annotations

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.random.state import _dispatch_random


def loggamma(*args: Tensor | float | int | str, **kwargs: Tensor | float | int | str) -> Tensor:
    """Evaluate loggamma operation.

    Args:
        *args (Tensor | float | int | str): Positional args.
        **kwargs (Tensor | float | int | str): Keyword args.

    Returns:
        Tensor: Result.
    """
    return _dispatch_random("loggamma", *args, **kwargs)
