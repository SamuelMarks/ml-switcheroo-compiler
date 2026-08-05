"""Core abstractions and logic definitions for laplace.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def laplace(*args: object, **kwargs: object) -> object:
    """Evaluate laplace operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return _dispatch_random("laplace", *args, **kwargs)
