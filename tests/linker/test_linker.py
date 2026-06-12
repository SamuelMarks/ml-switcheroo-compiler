"""Unit tests for the AST source linker backend.

This module contains test cases to verify the behavior of `get_source_ast_ref`, ensuring
it correctly retrieves file and line number references from the call stack under normal,
edge, and error conditions.
"""

from typing import NoReturn

from ml_switcheroo.backends.linker import get_source_ast_ref


def my_caller() -> object:
    """Calls `get_source_ast_ref` to simulate an additional call stack frame.

    Returns:
    object: The source AST reference string or None if it cannot be resolved.
    """
    return get_source_ast_ref()


def test_get_source_ast_ref() -> None:
    """Tests the standard behavior and frame depth traversal of `get_source_ast_ref`.

    Returns:
    None
    """
    ref = my_caller()
    assert "test_linker.py" in ref
    assert ":" in ref

    # Check depth
    ref2 = get_source_ast_ref(back_frames=0)
    assert "test_linker.py" in ref2


def test_linker_edge_cases() -> None:
    """Tests edge cases for `get_source_ast_ref`.

    This includes scenarios where the call stack frame cannot be retrieved
    (e.g., `sys._getframe` returns None) or when an excessively high number
    of back frames is requested

    Returns:
    None
    """
    import sys

    # Mock currentframe returning None
    old_currentframe = sys._getframe
    try:
        sys._getframe = lambda *args, **kwargs: None
        assert get_source_ast_ref() is None
    finally:
        sys._getframe = old_currentframe

    # High back frames
    assert get_source_ast_ref(back_frames=100) is not None


def test_linker_exception(monkeypatch: object) -> None:
    """Tests that `get_source_ast_ref` handles internal exceptions gracefully.

    Verifies that the function returns None when `inspect.getframeinfo`
    raises an exception

    Args:
    monkeypatch (object): The pytest monkeypatch fixture used to mock
        `inspect.getframeinfo`

    Returns:
    None
    """
    import inspect

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
