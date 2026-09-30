"""Tests for test_types_registry_coverage."""

from __future__ import annotations

import ml_switcheroo_compiler.diagnostics.types_registry as types_reg_mod


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


def test_types_registry_exhaustive() -> None:
    """Verify loading, parsing, fallback, and caching in types_registry.py."""
    # 1. Non-existent YAML path
    empty_cfg = types_reg_mod.load_types_registry("/path/does/not/exist/types.yaml")
    assert empty_cfg.types == {}

    # 2. Existing default YAML path
    default_cfg = types_reg_mod.load_types_registry(None)
    assert len(default_cfg.types) > 0
    assert "AcceleratorError" in default_cfg.types

    # 3. is_non_math_type cache initialization and lookup
    types_reg_mod._TYPES_CACHE = None
    assert types_reg_mod.is_non_math_type("AcceleratorError") is True
    # Cached hit branch
    assert types_reg_mod.is_non_math_type("AcceleratorError") is True
    assert types_reg_mod.is_non_math_type("DefinitelyNotAClass_12345") is False
