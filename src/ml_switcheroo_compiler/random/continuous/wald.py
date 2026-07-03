"""Module docstring."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def wald(*args: object, **kwargs: object) -> object:
    """Execute wald."""
    return _dispatch_random("wald", *args, **kwargs)
