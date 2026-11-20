"""Unit tests for core compiler error hierarchy, metadata formatting, and template loading."""

from __future__ import annotations

import os

import pytest

import ml_switcheroo_compiler.core.errors as err_mod
from ml_switcheroo_compiler.core.errors import (
    AutodiffError,
    AutodiffLinkageError,
    BackendDecouplingError,
    BackendNotSupportedError,
    CompilationError,
    CompilerError,
    DependencyViolationError,
    DTypePromotionError,
    DTypeValidationError,
    EvaluationError,
    FrontendBoundaryViolationError,
    MissingJVPRuleError,
    ShapeInferenceError,
    ShapeMismatchError,
    SnapshotGroundingError,
    SwitcherooError,
    SymbolicDifferentiationError,
    TracingError,
    UnimplementedMathError,
)


def test_error_subclasses_and_context_metadata() -> None:
    """Verify all compiler error classes inherit from CompilerError and format diagnostic metadata."""
    assert SwitcherooError is CompilerError

    base_err = SwitcherooError("Direct message")
    assert str(base_err) == "Direct message"

    ctx_err = CompilerError(
        "Ctx msg",
        node_id="n1",
        op_type="Add",
        backend_target="numpy",
        trace_context="line 1",
    )
    rendered = str(ctx_err)
    assert rendered == "Ctx msg | Node ID: n1 | Op Type: Add | Backend: numpy | Context: line 1"
    assert ctx_err.node_id == "n1"
    assert ctx_err.op_type == "Add"
    assert ctx_err.backend_target == "numpy"
    assert ctx_err.trace_context == "line 1"

    error_classes: list[type[CompilerError]] = [
        FrontendBoundaryViolationError,
        BackendDecouplingError,
        DependencyViolationError,
        UnimplementedMathError,
        AutodiffError,
        ShapeInferenceError,
        SnapshotGroundingError,
        TracingError,
        CompilationError,
        ShapeMismatchError,
        DTypePromotionError,
        DTypeValidationError,
        BackendNotSupportedError,
        MissingJVPRuleError,
        EvaluationError,
        AutodiffLinkageError,
        SymbolicDifferentiationError,
    ]
    for cls in error_classes:
        inst = cls("Test error")
        assert isinstance(inst, CompilerError)
        assert str(inst) == "Test error"


def test_error_templates_from_yaml_and_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify real YAML template loading, keyword formatting, and KeyError/message fallbacks.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture for resetting cached templates.
    """
    monkeypatch.setattr(err_mod, "_ERROR_TEMPLATES", {})

    # Real error_messages.yaml reload and formatting
    err_shape = ShapeMismatchError(shape1="[2, 3]", shape2="[2, 4]")
    assert str(err_shape) == "Shape mismatch during static inference: [2, 3] != [2, 4]"

    # Cached _ERROR_TEMPLATES branch with missing template key falls back to unformatted template
    err_missing_kw = ShapeMismatchError(shape1="[2, 3]")
    assert str(err_missing_kw) == "Shape mismatch during static inference: {shape1} != {shape2}"

    # Explicit message takes precedence over template kwargs
    err_explicit = ShapeMismatchError(message="explicit override", shape1="[1]", shape2="[2]")
    assert str(err_explicit) == "explicit override"

    # Error class without its own entry falls back to SwitcherooError template if present
    monkeypatch.setattr(err_mod, "_ERROR_TEMPLATES", {"SwitcherooError": "Generic: {detail}"})
    err_fallback = SwitcherooError(detail="bad_input")
    assert str(err_fallback) == "Generic: bad_input"


def test_error_templates_when_yaml_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CompilerError gracefully handles missing error_messages.yaml file.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture for simulating missing YAML file.
    """
    monkeypatch.setattr(err_mod, "_ERROR_TEMPLATES", {})
    monkeypatch.setattr(os.path, "exists", lambda _path: False)

    err = SwitcherooError("no yaml message")
    assert str(err) == "no yaml message"
    assert err_mod._ERROR_TEMPLATES == {}
