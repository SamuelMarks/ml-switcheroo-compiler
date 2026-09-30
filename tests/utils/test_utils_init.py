"""Tests for ml_switcheroo_compiler.utils.__init__."""

from __future__ import annotations

import ml_switcheroo_compiler.utils as utils_mod


def test_utils_init() -> None:
    """Test importing ml_switcheroo_compiler.utils module."""
    assert utils_mod is not None
