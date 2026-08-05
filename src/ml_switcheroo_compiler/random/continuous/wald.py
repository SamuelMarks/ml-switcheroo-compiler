"""Core abstractions and logic definitions for wald.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def wald(*args: object, **kwargs: object) -> object:
    """Evaluate wald operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return _dispatch_random("wald", *args, **kwargs)
