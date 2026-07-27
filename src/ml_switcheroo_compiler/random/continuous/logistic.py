"""Core abstractions and logic definitions for logistic.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def logistic(*args: object, **kwargs: object) -> object:
    """Execute logistic."""
    return _dispatch_random("logistic", *args, **kwargs)
