"""Module docstring."""

from typing import Any

from ml_switcheroo.linker import get_source_ast_ref
from typing import NoReturn


def my_caller() -> Any:
    """Docstring."""
    return get_source_ast_ref()


def test_get_source_ast_ref() -> None:
    """Docstring."""
    ref = my_caller()
    assert "test_linker.py" in ref
    assert ":" in ref

    # Check depth
    ref2 = get_source_ast_ref(back_frames=0)
    assert "test_linker.py" in ref2


def test_linker_edge_cases() -> None:
    """Docstring."""
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


def test_linker_exception(monkeypatch: Any) -> None:
    """Docstring."""
    import inspect

    def mock_getframeinfo(*args: Any, **kwargs: Any) -> NoReturn:
        """Docstring."""
        raise ValueError("test error")

    monkeypatch.setattr(inspect, "getframeinfo", mock_getframeinfo)
    assert get_source_ast_ref() is None
