"""Core abstractions and logic definitions for maxwell.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def maxwell(*args: object, **kwargs: object) -> object:
    """Evaluate maxwell operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return _dispatch_random("maxwell", *args, **kwargs)
