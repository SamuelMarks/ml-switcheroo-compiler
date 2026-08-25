"""Module exponential.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for exponential.py."""

from ml_switcheroo_compiler.random.state import _dispatch_random


def exponential(*args: object, **kwargs: object) -> object:
    """Evaluate exponential operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _dispatch_random("exponential", *args, **kwargs)
