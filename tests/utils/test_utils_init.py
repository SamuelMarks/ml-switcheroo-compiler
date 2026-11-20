"""Unit tests for `ml_switcheroo_compiler.utils` package and filesystem helpers."""

from __future__ import annotations

import ml_switcheroo_compiler.utils as utils_mod
from ml_switcheroo_compiler.utils import file_utils


def test_utils_init() -> None:
    """Verify importing `ml_switcheroo_compiler.utils` exposes a valid module."""
    assert type(utils_mod).__name__ == "module"


def test_file_utils_exists() -> None:
    """Verify `file_utils.exists` accurately checks path existence on disk."""
    assert file_utils.exists(__file__) is True
    assert file_utils.exists("non_existent_file_path_12345.xyz") is False
