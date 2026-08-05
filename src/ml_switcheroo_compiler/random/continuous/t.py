"""Core abstractions and logic definitions for t.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def t(*args: object, **kwargs: object) -> object:
    """Evaluate t operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return _dispatch_random("t", *args, **kwargs)
