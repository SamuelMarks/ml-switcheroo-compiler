"""Unit tests verifying operator data type validation and rejection of invalid types."""

from __future__ import annotations

import pytest

from ml_switcheroo_compiler.core.dtype import (
    DType,
    DTypeValidationError,
    check_dtype_for_op,
    validate_dtype_for_op,
)


def test_transcendental_ops_accept_floats_and_complex() -> None:
    """Verify that transcendental and decomposition operations accept float and complex dtypes."""
    valid_dtypes: list[DType] = [
        DType.Float32,
        DType.Float64,
        DType.Float16,
        DType.BFloat16,
        DType.Complex64,
        DType.Complex128,
    ]
    ops: list[str] = ["sin", "cos", "exp", "log", "cholesky", "inv", "sqrt", "sigmoid"]
    for op in ops:
        for dt in valid_dtypes:
            is_valid, err = validate_dtype_for_op(op, dt)
            assert is_valid is True, f"Expected {dt} to be valid for {op}"
            assert err is None
            check_dtype_for_op(op, dt)


def test_transcendental_ops_reject_integers_and_bools() -> None:
    """Verify that transcendental and decomposition operations reject int, uint, and bool dtypes."""
    invalid_dtypes: list[DType] = [
        DType.Int32,
        DType.Int64,
        DType.Int16,
        DType.Int8,
        DType.UInt8,
        DType.UInt32,
        DType.Bool,
        DType.String,
    ]
    ops: list[str] = ["sin", "cos", "exp", "log", "cholesky", "linalg_inv", "inv"]
    for op in ops:
        for dt in invalid_dtypes:
            is_valid, err = validate_dtype_for_op(op, dt)
            assert is_valid is False, f"Expected {dt} to be invalid for {op}"
            assert err is not None
            assert "not supported for transcendental" in err

            with pytest.raises(DTypeValidationError) as exc_info:
                check_dtype_for_op(op, dt)
            assert "not supported for transcendental" in str(exc_info.value)


def test_general_ops_accept_integers() -> None:
    """Verify that non-transcendental operations accept integer and numeric types."""
    ops: list[str] = ["add", "multiply", "subtract", "reshape", "concat"]
    for op in ops:
        is_valid, err = validate_dtype_for_op(op, DType.Int32)
        assert is_valid is True
        assert err is None
        check_dtype_for_op(op, DType.Int32)


def test_validate_dtype_fallback_and_string_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify validate_dtype_for_op fallback behavior when compliance import fails.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    import importlib

    real_import = importlib.import_module

    def mock_import(name: str, package: str | None = None) -> object:
        if "ml_ecosystem_snapshots" in name:
            raise ImportError("Simulated missing module")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", mock_import)

    # Test string dtype and fallback logic
    valid, err = validate_dtype_for_op("sin", "float32")
    assert valid is True
    assert err is None

    invalid, err_msg = validate_dtype_for_op("sin", "int32")
    assert invalid is False
    assert err_msg is not None
    assert "not supported for transcendental" in err_msg

    # Test non-transcendental op fallback
    valid_op, err_op = validate_dtype_for_op("add", "int32")
    assert valid_op is True
    assert err_op is None


def test_validate_dtype_fn_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify fallback behavior when compliance module exists but function attribute is None.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    import importlib

    real_import = importlib.import_module

    class DummyMod:
        validate_dtype_for_op = None

    def mock_import(name: str, package: str | None = None) -> object:
        if "ml_ecosystem_snapshots" in name:
            return DummyMod()
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", mock_import)
    valid, err = validate_dtype_for_op("sin", "float32")
    assert valid is True
    assert err is None
