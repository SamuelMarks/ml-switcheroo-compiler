"""Core abstractions and logic definitions for pareto.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def pareto(*args: object, **kwargs: object) -> object:
    """Evaluate pareto operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return _dispatch_random("pareto", *args, **kwargs)
