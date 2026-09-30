"""Tests for test_file_utils_coverage."""

from __future__ import annotations

import ml_switcheroo_compiler.utils.file_utils as file_mod


def test_file_utils_exists() -> None:
    """Verify file_utils.exists accurately checks path existence."""
    assert file_mod.exists(__file__) is True
    assert file_mod.exists("non_existent_file_path_12345.xyz") is False
