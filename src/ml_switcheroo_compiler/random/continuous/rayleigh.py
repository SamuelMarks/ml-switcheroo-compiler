"""Module docstring."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def rayleigh(*args: object, **kwargs: object) -> object:
    """Execute rayleigh."""
    return _dispatch_random("rayleigh", *args, **kwargs)
