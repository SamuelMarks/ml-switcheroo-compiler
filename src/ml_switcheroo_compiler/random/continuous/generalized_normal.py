"""Core abstractions and logic definitions for generalized_normal.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def generalized_normal(*args: object, **kwargs: object) -> object:
    """Evaluate generalized_normal operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return _dispatch_random("generalized_normal", *args, **kwargs)
