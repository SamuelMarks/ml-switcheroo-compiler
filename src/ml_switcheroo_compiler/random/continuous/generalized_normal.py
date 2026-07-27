"""Core abstractions and logic definitions for generalized_normal.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def generalized_normal(*args: object, **kwargs: object) -> object:
    """Execute generalized_normal."""
    return _dispatch_random("generalized_normal", *args, **kwargs)
