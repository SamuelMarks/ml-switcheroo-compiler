"""Module docstring."""

import importlib


def test_import_error_branches() -> None:
    """Docstring."""
    import ml_switcheroo_compiler.backends as be

    # Force reload of the module while simulating ImportError
    orig_import = __import__

    def mock_import(
        name: str, globals: dict = None, locals: dict = None, fromlist: tuple = (), level: int = 0
    ) -> object:
        """Docstring."""
        if name in ("ml_switcheroo_compiler.backends.cupy", "ml_switcheroo_compiler.backends.dask"):
            raise ImportError(f"Mocked ImportError for {name}")
        return orig_import(name, globals, locals, fromlist, level)

    import builtins

    builtins.__import__ = mock_import
    try:
        importlib.reload(be)
    finally:
        builtins.__import__ = orig_import
