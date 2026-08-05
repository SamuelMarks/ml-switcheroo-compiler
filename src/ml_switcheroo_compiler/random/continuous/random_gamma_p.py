"""Core abstractions and logic definitions for random_gamma_p.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def random_gamma_p(*args: object, **kwargs: object) -> object:
    """Evaluate random_gamma_p operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return _dispatch_random("random_gamma_p", *args, **kwargs)
