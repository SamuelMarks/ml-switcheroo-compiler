"""Compiler `random gamma p` operations and primitives for the `continuous` subsystem."""

from __future__ import annotations

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.random.state import _dispatch_random


def random_gamma_p(*args: Tensor | float | int | str, **kwargs: Tensor | float | int | str) -> Tensor:
    """Execute random gamma p for compiler processing.

    Args:
        *args (Tensor | float | int | str): Positional args.
        **kwargs (Tensor | float | int | str): Keyword args.

    Returns:
        Tensor: Computed `Tensor` output.
    """
    return _dispatch_random("random_gamma_p", *args, **kwargs)
