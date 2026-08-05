"""Core abstractions and logic definitions for exponential.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def exponential(*args: object, **kwargs: object) -> object:
    """Evaluate exponential operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return _dispatch_random("exponential", *args, **kwargs)
