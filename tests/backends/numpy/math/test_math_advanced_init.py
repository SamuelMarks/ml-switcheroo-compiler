"""Tests for test_math_advanced_init."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.eager import reference_math as math_advanced_mod
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry, numpy_eager_registry


def test_math_advanced_init() -> None:
    """Test reference_math __init__ exports and registrations.

    Returns:
        None
    """
    assert math_advanced_mod.__all__ is not None
    assert "math_general" in math_advanced_mod.__all__
    assert global_eager_registry.get("rem") is not None
    assert numpy_eager_registry.get("rem") is not None
    assert numpy_eager_registry.get("descriptive") is not None
