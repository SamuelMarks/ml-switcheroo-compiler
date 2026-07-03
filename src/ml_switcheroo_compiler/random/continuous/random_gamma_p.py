"""Module docstring."""

from __future__ import annotations

from ml_switcheroo_compiler.random.state import _dispatch_random


def random_gamma_p(*args: object, **kwargs: object) -> object:
    """Execute random_gamma_p."""
    return _dispatch_random("random_gamma_p", *args, **kwargs)
