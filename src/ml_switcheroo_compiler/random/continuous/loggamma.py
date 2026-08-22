"""Module loggamma.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for loggamma.py."""
from typing import Any

from ml_switcheroo_compiler.random.state import _dispatch_random


def loggamma(*args: Any, **kwargs: Any) -> Any:
    """Evaluate loggamma operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _dispatch_random("loggamma", *args, **kwargs)
