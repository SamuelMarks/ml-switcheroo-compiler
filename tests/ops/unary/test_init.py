"""Tests for test_init."""

from __future__ import annotations


def test_unary_init() -> None:
    """Test function test_unary_init."""
    import ml_switcheroo_compiler.ops.unary as unary

    assert unary.abs is not None
