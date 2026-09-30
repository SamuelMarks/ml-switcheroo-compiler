"""Tests for test_errors_coverage."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import ml_switcheroo_compiler.core.errors as err_mod
from ml_switcheroo_compiler.core.errors import (
    BackendNotSupportedError,
    CompilationError,
    DTypePromotionError,
    MissingJVPRuleError,
    ShapeMismatchError,
    SwitcherooError,
    TracingError,
    UnimplementedMathError,
)


def test_core_errors_exhaustive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify SwitcherooError formatting, templates, KeyError fallback, and all subclasses."""
    # 1. Base error with custom message
    e = SwitcherooError("Direct message")
    assert str(e) == "Direct message"

    # 2. Template formatting with kwargs
    monkeypatch.setattr(err_mod, "_ERROR_TEMPLATES", {"ShapeMismatchError": "Expected {expected}, got {got}"})
    err_formatted = ShapeMismatchError(expected="[2, 3]", got="[2, 4]")
    assert str(err_formatted) == "Expected [2, 3], got [2, 4]"

    # 3. Template formatting KeyError fallback
    err_missing_key = ShapeMismatchError(expected="[2, 3]")  # 'got' is missing
    assert str(err_missing_key) == "Expected {expected}, got {got}"

    # 4. _load_error_templates when file missing vs exists
    monkeypatch.setattr(err_mod, "_ERROR_TEMPLATES", {})
    with patch("os.path.exists", return_value=False):
        err_mod._load_error_templates()
        assert err_mod._ERROR_TEMPLATES == {}

    # 5. Verify all error subclasses
    classes = [
        TracingError,
        CompilationError,
        DTypePromotionError,
        BackendNotSupportedError,
        UnimplementedMathError,
        MissingJVPRuleError,
    ]
    for cls in classes:
        inst = cls("Test error")
        assert isinstance(inst, SwitcherooError)
        assert str(inst) == "Test error"
