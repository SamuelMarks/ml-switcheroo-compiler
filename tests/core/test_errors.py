"""Tests for core errors."""

import os
import tempfile
from unittest import mock

import yaml

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
        MissingJVPRuleError("test"),
    ]

    for error in errors:
        assert isinstance(error, Exception)
        if type(error) is not SwitcherooError:
            assert isinstance(error, SwitcherooError)
        assert str(error) == "test"


def test_error_templates() -> None:
    """Test that error templates are loaded and used correctly."""
    # Mock os.path.exists and open to return a temporary yaml file
    yaml_content = {"SwitcherooError": "Error: {arg}"}

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        yaml.dump(yaml_content, f)
        f.flush()
        f.close()
        temp_file_name = f.name

    with mock.patch("ml_switcheroo_compiler.core.errors._ERROR_TEMPLATES", {}), mock.patch("os.path.join", return_value=temp_file_name):
        # Should use template
        err1 = SwitcherooError(arg="test_arg")
        assert str(err1) == "Error: test_arg"

        # Should fall back to passed message if missing key
        err2 = SwitcherooError(message="fallback message", missing_arg="val")
        assert str(err2) == "fallback message"

        # Test no kwargs falls back
        err3 = SwitcherooError("no kwargs message")
        assert str(err3) == "no kwargs message"

    os.remove(temp_file_name)


def test_load_error_templates_no_file() -> None:
    """Test loading templates when file does not exist."""
    with mock.patch("ml_switcheroo_compiler.core.errors._ERROR_TEMPLATES", {}), mock.patch("os.path.exists", return_value=False):
        err = SwitcherooError("message")
        assert str(err) == "message"
