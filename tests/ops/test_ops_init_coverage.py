"""Module docstring."""

from ml_switcheroo_compiler.ops import clamp
from ml_switcheroo_compiler.core.config import config


def test_clamp_coverage_brute() -> None:
    """Function docstring."""
    config.eager_mode = True
    clamp(None, 1, 10)
    clamp(0, 1, None)
    config.eager_mode = False
