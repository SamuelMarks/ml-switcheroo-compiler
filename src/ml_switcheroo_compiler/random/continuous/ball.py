"""Core abstractions and logic definitions for ball.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def ball(*args: object, **kwargs: object) -> object:
    """Evaluate ball operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return _dispatch_random("ball", *args, **kwargs)
