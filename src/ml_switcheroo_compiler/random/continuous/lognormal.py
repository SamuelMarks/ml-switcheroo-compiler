"""Core abstractions and logic definitions for lognormal.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def lognormal(*args: object, **kwargs: object) -> object:
    """Evaluate lognormal operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return _dispatch_random("lognormal", *args, **kwargs)
