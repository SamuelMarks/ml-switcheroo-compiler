"""Module docstring."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def chisquare(*args: object, **kwargs: object) -> object:
    """Execute chisquare."""
    return _dispatch_random("chisquare", *args, **kwargs)
