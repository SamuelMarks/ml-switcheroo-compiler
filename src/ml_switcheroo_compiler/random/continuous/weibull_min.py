"""Compiler `weibull min` operations and primitives for the `continuous` subsystem."""

from __future__ import annotations

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.random.state import _dispatch_random


def weibull_min(*args: Tensor | float | int | str, **kwargs: Tensor | float | int | str) -> Tensor:
    """Execute weibull min for compiler processing.

    Args:
        *args (Tensor | float | int | str): Positional args.
        **kwargs (Tensor | float | int | str): Keyword args.

    Returns:
        Tensor: Computed `Tensor` output.
    """
    return _dispatch_random("weibull_min", *args, **kwargs)
