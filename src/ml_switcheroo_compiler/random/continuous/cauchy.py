"""Core abstractions and logic definitions for cauchy.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def cauchy(*args: object, **kwargs: object) -> object:
    """Evaluate cauchy operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return _dispatch_random("cauchy", *args, **kwargs)
