"""Module docstring."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def gumbel(*args: object, **kwargs: object) -> object:
    """Execute gumbel."""
    return _dispatch_random("gumbel", *args, **kwargs)
