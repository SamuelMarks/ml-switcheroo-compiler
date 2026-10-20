"""Unit tests for the AST source linker backend.

This module contains test cases to verify the behavior of `get_source_ast_ref`, ensuring
it correctly retrieves file and line number references from the call stack under normal,
edge, and error conditions.
"""

from __future__ import annotations

import inspect
import sys
import typing
from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.backends.linker import get_source_ast_ref

if typing.TYPE_CHECKING:
    import pytest

    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def my_caller() -> EagerValue:
    """Call `get_source_ast_ref` to simulate an additional call stack frame.

    Returns:
        EagerValue: The source AST reference string or None.
    """
    return get_source_ast_ref()


def test_get_source_ast_ref() -> None:
    """Test standard behavior and frame depth traversal of `get_source_ast_ref`."""
    ref = my_caller()
    assert isinstance(ref, str)
    assert "test_linker.py" in ref
    assert ":" in ref
    ref2 = get_source_ast_ref(back_frames=0)
    assert isinstance(ref2, str)
    assert "test_linker.py" in ref2


def test_linker_edge_cases() -> None:
    """Test edge cases for `get_source_ast_ref` when frame retrieval fails or overflows."""
    old_currentframe = sys._getframe
    try:
        setattr(sys, "_getframe", lambda *_args, **_kwargs: None)
        assert get_source_ast_ref() is None
    finally:
        sys._getframe = old_currentframe
    ref_deep = get_source_ast_ref(back_frames=100)
    assert isinstance(ref_deep, str)
    assert ":" in ref_deep


def test_linker_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that `get_source_ast_ref` returns None when `inspect.getframeinfo` raises ValueError.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """

    def mock_getframeinfo(*_args: EagerValue, **_kwargs: EagerValue) -> typing.NoReturn:
        """Raise ValueError to simulate an inspection failure.

        Args:
            *_args (EagerValue): Positional arguments passed to the mock.
            **_kwargs (EagerValue): Keyword arguments passed to the mock.

        Raises:
            ValueError: Always raised to simulate an inspection error.
        """
        msg = "test error"
        raise ValueError(msg)

    monkeypatch.setattr(inspect, "getframeinfo", mock_getframeinfo)
    assert get_source_ast_ref() is None


def test_backends_linker_edge_cases() -> None:
    """Test `get_source_ast_ref` edge cases with mocked `inspect.currentframe`."""
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
