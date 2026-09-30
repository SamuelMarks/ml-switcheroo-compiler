"""Tests for test_math_advanced_init."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import math_advanced as math_advanced_mod


def test_math_advanced_init() -> None:
    """Test math_advanced __init__ exports and registrations.

    Returns:
        None
    """
    assert hasattr(math_advanced_mod, "__all__")
    assert "numpy_eager_registry" in math_advanced_mod.__all__
    assert numpy_eager_registry.get("rem") is not None
    assert numpy_eager_registry.get("descriptive") is not None
