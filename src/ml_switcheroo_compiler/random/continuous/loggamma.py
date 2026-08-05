"""Core abstractions and logic definitions for loggamma.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def loggamma(*args: object, **kwargs: object) -> object:
    """Evaluate loggamma operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return _dispatch_random("loggamma", *args, **kwargs)
