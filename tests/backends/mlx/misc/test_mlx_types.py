"""Test mlx coverage."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.backends.mlx.types import array, asarray, item, zeros
from tests.testing_utils import adapt_backend_module

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg

pytest.importorskip("mlx.core", exc_type=ImportError)


def test_mlx_types() -> None:
    """Test mlx types."""
    assert array.__name__ == "array"
    assert asarray.__name__ == "asarray"
    assert item.__name__ == "item"
    assert zeros.__name__ == "zeros"
    import mlx.core as mx

    array(None, [1, 2])
    asarray(None, [3, 4])
    zeros(None, (2,))
    item(None, mx.array([5]))


def test_mlx_types_asarray_mock() -> None:
    """Test mlx asarray mock."""
    import mlx.core as mx

    from ml_switcheroo_compiler.backends.mlx.types import asarray

    pytest.MonkeyPatch().setattr(mx, "asarray", lambda x: mx.array(x), raising=False)
    asarray(None, [1, 2])
    pytest.MonkeyPatch().delattr(mx, "asarray")
    assert callable(asarray)


def test_mlx_init_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_mlx_init_import_error.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    import importlib.util
    import sys

    # Force the module to be reloaded without mlx
    monkeypatch.delitem(sys.modules, "ml_switcheroo_compiler.backends.mlx", raising=False)

    def mock_find_spec(name: BackendArg, package: BackendArg = None) -> BackendArg:
        """Test function mock_find_spec.

        Args:
            name: Test parameter `name`.
            package: Test parameter `package`.

        Returns:
            BackendArg: Computed output value from `mock_find_spec`.
        """
        if name == "mlx":
            return None
        return adapt_backend_module(importlib.util).require_symbol("_find_spec")(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    monkeypatch.delitem(sys.modules, "sphinx", raising=False)

    with pytest.raises(ImportError, match="requires the 'mlx' library to be installed"):
        importlib.import_module("ml_switcheroo_compiler.backends.mlx")
