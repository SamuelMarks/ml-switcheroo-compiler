"""Core abstractions and logic definitions for cauchy.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def cauchy(*args: object, **kwargs: object) -> object:
    """Execute cauchy."""
    return _dispatch_random("cauchy", *args, **kwargs)
