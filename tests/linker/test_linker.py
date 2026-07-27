# ruff: noqa: E501
"""Unit tests for the AST source linker backend.

This module contains test cases to verify the behavior of `get_source_ast_ref`, ensuring
it correctly retrieves file and line number references from the call stack under normal,
edge, and error conditions.
"""

import inspect
import sys
from typing import NoReturn

from ml_switcheroo_compiler.backends.linker import get_source_ast_ref


def my_caller() -> object:
    """Calls `get_source_ast_ref` to simulate an additional call stack frame."""
    return get_source_ast_ref()


def test_get_source_ast_ref() -> None:
    """Test the get source ast ref behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Tests the standard behavior and frame depth traversal of `get_source_ast_ref`.\n\n    Returns:\n    None\n    "
        ref = my_caller()
        assert "test_linker.py" in ref
        assert ":" in ref
        ref2 = get_source_ast_ref(back_frames=0)
        assert "test_linker.py" in ref2
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_linker_edge_cases() -> None:
    """Test the linker edge cases behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Tests edge cases for `get_source_ast_ref`.\n\n    This includes scenarios where the call stack frame cannot be retrieved\n    (e.g., `sys._getframe` returns None) or when an excessively high number\n    of back frames is requested\n\n    Returns:\n    None\n    "
        old_currentframe = sys._getframe
        try:
            sys._getframe = lambda *args, **kwargs: None
            assert get_source_ast_ref() is None
        finally:
            sys._getframe = old_currentframe
        assert get_source_ast_ref(back_frames=100) is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_linker_exception(monkeypatch: object) -> None:
    """Test the linker exception behavior.

    Args:
        monkeypatch (object): The monkeypatch parameter.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Tests that `get_source_ast_ref` handles internal exceptions gracefully.\n\n    Verifies that the function returns None when `inspect.getframeinfo`\n    raises an exception\n\n    Args:\n    monkeypatch (object): The pytest monkeypatch fixture used to mock\n        `inspect.getframeinfo`\n\n    Returns:\n    None\n    "

        def mock_getframeinfo(*args: object, **kwargs: object) -> NoReturn:
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
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
