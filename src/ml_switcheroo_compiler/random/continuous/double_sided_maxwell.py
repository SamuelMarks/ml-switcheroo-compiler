"""Core abstractions and logic definitions for double_sided_maxwell.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def double_sided_maxwell(*args: object, **kwargs: object) -> object:
    """Execute double_sided_maxwell."""
    return _dispatch_random("double_sided_maxwell", *args, **kwargs)
