"""Unit tests for diagnostics types registry and non-math metadata pruning."""

from __future__ import annotations

import pytest

import ml_switcheroo_compiler.diagnostics.types_registry as types_reg_mod
from ml_switcheroo_compiler.diagnostics.types_registry import (
    is_non_math_type,
    load_types_registry,
)
from ml_switcheroo_compiler.ops.registry import get_op


def test_types_registry_loading() -> None:
    """Verify loading default types registry and handling missing paths."""
    registry = load_types_registry()
    assert len(registry.types) > 0
    assert "AcceleratorError" in registry.types
    assert "Arch" in registry.types
    assert "AdaptiveGradClipState" in registry.types

    missing_reg = load_types_registry("/nonexistent/file/path.yaml")
    assert len(missing_reg.types) == 0


def test_is_non_math_type() -> None:
    """Verify non-math type discrimination against genuine mathematical ops."""
    assert is_non_math_type("AcceleratorError") is True
    assert is_non_math_type("Arch") is True
    assert is_non_math_type("AdaptiveGradClipState") is True
    assert is_non_math_type("Argument") is True

    assert is_non_math_type("Add") is False
    assert is_non_math_type("Conv2D") is False
    assert is_non_math_type("MatMul") is False
    assert is_non_math_type("Relu") is False


def test_get_op_rejects_non_math_types() -> None:
    """Verify primary math execution dispatcher rejects non-math empty stubs."""
    with pytest.raises(KeyError, match="non-math metadata type/exception"):
        get_op("AcceleratorError")

    with pytest.raises(KeyError, match="non-math metadata type/exception"):
        get_op("Arch")

    with pytest.raises(KeyError, match="non-math metadata type/exception"):
        get_op("AdaptiveGradClipState")

    # Genuine op succeeds
    op_cls = get_op("Abs")
    assert op_cls is not None


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

        Returns:
            int: Computed output value from `__next__`.
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
