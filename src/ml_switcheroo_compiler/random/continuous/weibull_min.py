"""Core abstractions and logic definitions for weibull_min.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def weibull_min(*args: object, **kwargs: object) -> object:
    """Evaluate weibull_min operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return _dispatch_random("weibull_min", *args, **kwargs)
