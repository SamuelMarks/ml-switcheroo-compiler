"""Test mlx coverage."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.backends.mlx.types import array, asarray, item, zeros
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue

pytest.importorskip("mlx.core", exc_type=ImportError)


def test_mlx_types() -> None:
    """Test mlx types."""
    assert array is not None
    assert asarray is not None
    assert item is not None
    assert zeros is not None
    import mlx.core as mx

    ensure_callable(array)(None, [1, 2])
    ensure_callable(asarray)(None, [3, 4])
    ensure_callable(zeros)(None, (2,))
    ensure_callable(item)(None, ensure_callable(mx.array)([5]))


def test_mlx_types_asarray_mock() -> None:
    """Test mlx asarray mock."""
    import mlx.core as mx

    from ml_switcheroo_compiler.backends.mlx.types import asarray

    ensure_mutable(mx).asarray = ensure_eager_tensor(lambda x: mx.array(x))
    ensure_callable(asarray)(None, [1, 2])
    del ensure_mutable(mx).asarray
    assert callable(asarray)


def test_mlx_init_import_error(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_mlx_init_import_error.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    import importlib.util
    import sys

    # Force the module to be reloaded without mlx
    monkeypatch.delitem(sys.modules, "ml_switcheroo_compiler.backends.mlx", raising=False)

    def mock_find_spec(name: EagerValue, package: EagerValue = None) -> EagerValue:
        """Test function mock_find_spec.

        Args:
            name: Test parameter `name`.
            package: Test parameter `package`.

        Returns:
            EagerValue: Computed output value from `mock_find_spec`.
        """
        if name == "mlx":
            return None
        return ensure_eager_tensor(ensure_eager_tensor(importlib.util)._find_spec(name, package))

    monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    monkeypatch.delitem(sys.modules, "sphinx", raising=False)

    with pytest.raises(ImportError, match="requires the 'mlx' library to be installed"):
        importlib.import_module("ml_switcheroo_compiler.backends.mlx")
