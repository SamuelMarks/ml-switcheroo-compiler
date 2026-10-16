"""Tests for test_activations."""

from __future__ import annotations

import ml_switcheroo_compiler.nn.activations as act


def test_activations_import() -> None:
    """Test function test_activations_import."""
    assert act is not None
