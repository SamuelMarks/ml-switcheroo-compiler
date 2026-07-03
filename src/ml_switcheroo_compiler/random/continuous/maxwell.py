"""Module docstring."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def maxwell(*args: object, **kwargs: object) -> object:
    """Execute maxwell."""
    return _dispatch_random("maxwell", *args, **kwargs)
