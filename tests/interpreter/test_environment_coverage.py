"""Tests for test_environment_coverage."""

from __future__ import annotations

import pytest

import ml_switcheroo_compiler.interpreter.environment as env_mod


class _DummyBadShape:
    """Shape object whose iterator raises TypeError."""

    def __iter__(self) -> _DummyBadShape:
        """Return self as iterator.

        Returns:
            _DummyBadShape: Self.
        """
        return self

    def __next__(self) -> int:
        """Raise TypeError on iteration.

        Raises:
            TypeError: Simulated invalid dimension.
        """
        raise TypeError("Not iterable dimension")


def test_interpreter_environment_exhaustive() -> None:
    """Verify Environment initialization, retrieval, update, and containment."""
    # 1. Default initialization
    env_default = env_mod.Environment()
    assert env_default.memory == {}

    # 2. Initial values
    env = env_mod.Environment({"var_x": 3.14, "dim_n": 128})
    assert "var_x" in env
    assert "missing_key" not in env
    assert env.get("var_x") == 3.14

    # 3. get missing raises ValueError
    with pytest.raises(ValueError, match="Missing input value for node 'not_found'"):
        env.get("not_found")

    # 4. set new value
    env.set("var_y", 42)
    assert env.get("var_y") == 42
    assert "var_y" in env
