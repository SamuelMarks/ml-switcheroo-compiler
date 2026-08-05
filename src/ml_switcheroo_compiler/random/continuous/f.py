"""Core abstractions and logic definitions for f.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def f(*args: object, **kwargs: object) -> object:
    """Evaluate f operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return _dispatch_random("f", *args, **kwargs)
