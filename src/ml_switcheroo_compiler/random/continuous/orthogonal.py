"""Core abstractions and logic definitions for orthogonal.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def orthogonal(*args: object, **kwargs: object) -> object:
    """Evaluate orthogonal operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return _dispatch_random("orthogonal", *args, **kwargs)
