"""Core abstractions and logic definitions for triangular.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def triangular(*args: object, **kwargs: object) -> object:
    """Execute triangular."""
    return _dispatch_random("triangular", *args, **kwargs)
