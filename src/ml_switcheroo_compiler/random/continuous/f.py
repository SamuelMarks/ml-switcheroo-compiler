"""Module docstring."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def f(*args: object, **kwargs: object) -> object:
    """Execute f."""
    return _dispatch_random("f", *args, **kwargs)
