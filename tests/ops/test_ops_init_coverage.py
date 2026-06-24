"""Provides required module functionality."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.aliases import clamp


def test_clamp_coverage_brute() -> None:
    """Execute the requested function."""
    config.eager_mode = True
    clamp(None, 1, 10)
    clamp(0, 1, None)
    config.eager_mode = False
