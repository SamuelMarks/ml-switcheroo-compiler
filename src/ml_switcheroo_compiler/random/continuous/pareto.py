"""Core abstractions and logic definitions for pareto.py."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def pareto(*args: object, **kwargs: object) -> object:
    """Execute pareto."""
    return _dispatch_random("pareto", *args, **kwargs)
