"""Module docstring."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def t(*args: object, **kwargs: object) -> object:
    """Execute t."""
    return _dispatch_random("t", *args, **kwargs)
