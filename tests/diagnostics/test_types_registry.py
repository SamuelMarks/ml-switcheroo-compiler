"""Unit tests for diagnostics types registry and non-math metadata pruning."""

import pytest

from ml_switcheroo_compiler.diagnostics.types_registry import (
    is_non_math_type,
    load_types_registry,
)
from ml_switcheroo_compiler.ops.registry import get_op


def test_types_registry_loading():
    """Verify loading default types registry and handling missing paths."""
    registry = load_types_registry()
    assert len(registry.types) > 0
    assert "AcceleratorError" in registry.types
    assert "Arch" in registry.types
    assert "AdaptiveGradClipState" in registry.types

    missing_reg = load_types_registry("/nonexistent/file/path.yaml")
    assert len(missing_reg.types) == 0


def test_is_non_math_type():
    """Verify non-math type discrimination against genuine mathematical ops."""
    assert is_non_math_type("AcceleratorError") is True
    assert is_non_math_type("Arch") is True
    assert is_non_math_type("AdaptiveGradClipState") is True
    assert is_non_math_type("Argument") is True

    assert is_non_math_type("Add") is False
    assert is_non_math_type("Conv2D") is False
    assert is_non_math_type("MatMul") is False
    assert is_non_math_type("Relu") is False


def test_get_op_rejects_non_math_types():
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
