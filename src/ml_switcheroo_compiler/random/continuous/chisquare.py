"""Core abstractions and logic definitions for chisquare.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def chisquare(*args: object, **kwargs: object) -> object:
    """Evaluate chisquare operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return _dispatch_random("chisquare", *args, **kwargs)
