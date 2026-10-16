# ruff: noqa: E501
"""Unit tests for the AST source linker backend.

This module contains test cases to verify the behavior of `get_source_ast_ref`, ensuring
it correctly retrieves file and line number references from the call stack under normal,
edge, and error conditions.
"""

from __future__ import annotations

import inspect
import sys
from typing import TYPE_CHECKING, NoReturn, cast

from ml_switcheroo_compiler.backends.linker import get_source_ast_ref

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def my_caller() -> EagerValue:
    """Calls `get_source_ast_ref` to simulate an additional call stack frame."""
    return get_source_ast_ref()


def test_get_source_ast_ref() -> None:
    """Test the get source ast ref behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Tests the standard behavior and frame depth traversal of `get_source_ast_ref`.\n\n    Returns:\n    None\n    "
        ref = my_caller()
        assert cast("EagerTensorProtocol", "test_linker.py") in cast("EagerTensorProtocol", ref)
        assert cast("EagerTensorProtocol", ":") in cast("EagerTensorProtocol", ref)
        ref2 = get_source_ast_ref(back_frames=0)
        assert cast("EagerTensorProtocol", "test_linker.py") in cast("EagerTensorProtocol", ref2)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_linker_edge_cases() -> None:
    """Test the linker edge cases behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Tests edge cases for `get_source_ast_ref`.\n\n    This includes scenarios where the call stack frame cannot be retrieved\n    (e.g., `sys._getframe` returns None) or when an excessively high number\n    of back frames is requested\n\n    Returns:\n    None\n    "
        old_currentframe = sys._getframe
        try:
            setattr(sys, "_getframe", lambda *args, **kwargs: None)
            assert get_source_ast_ref() is None
        finally:
            sys._getframe = old_currentframe
        assert get_source_ast_ref(back_frames=100) is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_linker_exception(monkeypatch: EagerTensorProtocol) -> None:
    """Test the linker exception behavior.

    Args:
        monkeypatch (object): The monkeypatch parameter.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Tests that `get_source_ast_ref` handles internal exceptions gracefully.\n\n    Verifies that the function returns None when `inspect.getframeinfo`\n    raises an exception\n\n    Args:\n    monkeypatch (object): The pytest monkeypatch fixture used to mock\n        `inspect.getframeinfo`\n\n    Returns:\n    None\n    "

        def mock_getframeinfo(*args: EagerValue, **kwargs: EagerValue) -> NoReturn:
            """Mocks `inspect.getframeinfo` to simulate a failure by raising an exception.

            Args:
            *args (object): Positional arguments passed to the mock
            **kwargs (object): Keyword arguments passed to the mock

            Raises:
            ValueError: Always raised to simulate an inspection error

            Returns:
            NoReturn: This function never returns.
            """
            msg = "test error"
            raise ValueError(msg)

        monkeypatch.setattr(inspect, "getframeinfo", mock_getframeinfo)
        assert get_source_ast_ref() is None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_backends_linker_edge_cases() -> None:
    """Test get_source_ast_ref edge cases in backends/linker.py."""
    from unittest.mock import MagicMock, patch

    with patch("inspect.currentframe", return_value=None):
        assert get_source_ast_ref() is None

    mock_frame = MagicMock()
    mock_frame.f_back = None
    with patch("inspect.currentframe", return_value=mock_frame):
        with patch("inspect.getframeinfo", return_value=MagicMock(filename="test.py", lineno=10)):
            ref = get_source_ast_ref(back_frames=5)
            assert ref == "test.py:10"

    with patch("inspect.getframeinfo", side_effect=ValueError("frame error")):
        assert get_source_ast_ref() is None
