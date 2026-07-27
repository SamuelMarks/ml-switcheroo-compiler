# ruff: noqa: D103
"""Tests for random extras."""

from ml_switcheroo_compiler.ops.random_stateless import create_rng_state, get_global_generator


def test_random_extras() -> None:
    # Just exercise them
    assert get_global_generator() is not None
    assert create_rng_state(0) is not None

    # Pragma no cover handles the rest.
