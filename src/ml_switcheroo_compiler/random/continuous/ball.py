"""Module docstring."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def ball(*args: object, **kwargs: object) -> object:
    """Execute ball."""
    return _dispatch_random("ball", *args, **kwargs)
