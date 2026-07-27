"""Core abstractions and logic definitions for laplace.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def laplace(*args: object, **kwargs: object) -> object:
    """Execute laplace."""
    return _dispatch_random("laplace", *args, **kwargs)
