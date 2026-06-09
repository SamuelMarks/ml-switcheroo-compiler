"""Module docstring."""

from ml_switcheroo_compiler.linker import get_source_ast_ref


def my_caller():
    """Docstring."""
    return get_source_ast_ref()


def test_get_source_ast_ref():
    """Docstring."""
    ref = my_caller()
    assert "test_linker.py" in ref
    assert ":" in ref

    # Check depth
    ref2 = get_source_ast_ref(back_frames=0)
    assert "test_linker.py" in ref2


def test_linker_edge_cases():
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


def test_linker_exception(monkeypatch):
    """Docstring."""
    import inspect

    def mock_getframeinfo(*args, **kwargs):
        """Docstring."""
        raise ValueError("test error")

    monkeypatch.setattr(inspect, "getframeinfo", mock_getframeinfo)
    assert get_source_ast_ref() is None
