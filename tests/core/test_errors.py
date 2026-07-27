"""Tests for core errors."""

from ml_switcheroo_compiler.core.errors import (
    BackendNotSupportedError,
    CompilationError,
    DTypePromotionError,
    ShapeMismatchError,
    SwitcherooError,
    TracingError,
    UnimplementedMathError,
)


def test_errors() -> None:
    """Test that all errors can be instantiated and inherit from SwitcherooError."""
    errors = [
        SwitcherooError("test"),
        TracingError("test"),
        CompilationError("test"),
        ShapeMismatchError("test"),
        DTypePromotionError("test"),
        BackendNotSupportedError("test"),
        UnimplementedMathError("test"),
    ]

    for error in errors:
        assert isinstance(error, Exception)
        if type(error) is not SwitcherooError:
            assert isinstance(error, SwitcherooError)
        assert str(error) == "test"
