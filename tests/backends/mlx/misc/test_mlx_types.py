"""Test mlx coverage."""

import pytest

pytest.importorskip("mlx")

from ml_switcheroo_compiler.backends.mlx.types import array, asarray, item, zeros


def test_mlx_types() -> None:
    """Test mlx types."""
    assert array is not None
    assert asarray is not None
    assert item is not None
    assert zeros is not None
    import mlx.core as mx

    array(None, [1, 2])
    asarray(None, [3, 4])
    zeros(None, (2,))
    item(None, mx.array([5]))


def test_mlx_types_asarray_mock() -> None:
    """Test mlx asarray mock."""
    import mlx.core as mx

    from ml_switcheroo_compiler.backends.mlx.types import asarray

    mx.asarray = lambda x: mx.array(x)
    asarray(None, [1, 2])
    del mx.asarray


def test_mlx_init_import_error(monkeypatch):
    import importlib.util
    import sys

    # Force the module to be reloaded without mlx
    if "ml_switcheroo_compiler.backends.mlx" in sys.modules:
        del sys.modules["ml_switcheroo_compiler.backends.mlx"]

    def mock_find_spec(name, package=None):
        if name == "mlx":
            return None
        return importlib.util._find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)

    import pytest

    with pytest.raises(ImportError, match="requires the 'mlx' library to be installed"):
        importlib.import_module("ml_switcheroo_compiler.backends.mlx")
