"""Test mlx coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from ml_switcheroo_compiler.backends.mlx.types import array, asarray, item, zeros

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue

pytest.importorskip("mlx.core")


def test_mlx_types() -> None:
    """Test mlx types."""
    assert array is not None
    assert asarray is not None
    assert item is not None
    assert zeros is not None
    import mlx.core as mx

    cast("EagerCallableProtocol", array)(None, [1, 2])
    cast("EagerCallableProtocol", asarray)(None, [3, 4])
    cast("EagerCallableProtocol", zeros)(None, (2,))
    cast("EagerCallableProtocol", item)(None, cast("EagerCallableProtocol", mx.array)([5]))


def test_mlx_types_asarray_mock() -> None:
    """Test mlx asarray mock."""
    import mlx.core as mx

    from ml_switcheroo_compiler.backends.mlx.types import asarray

    setattr(mx, "asarray", lambda x: mx.array(x))
    cast("EagerCallableProtocol", asarray)(None, [1, 2])
    delattr(mx, "asarray")
    assert callable(asarray)


def test_mlx_init_import_error(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_mlx_init_import_error.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    import importlib.util
    import sys

    # Force the module to be reloaded without mlx
    if "ml_switcheroo_compiler.backends.mlx" in sys.modules:
        del sys.modules["ml_switcheroo_compiler.backends.mlx"]

    def mock_find_spec(name: EagerValue, package: EagerValue = None) -> EagerValue:
        """Test function mock_find_spec.

        Args:
            name: Test parameter `name`.
            package: Test parameter `package`.
        """
        if name == "mlx":
            return None
        return cast("EagerValue", getattr(importlib.util, "_find_spec")(name, package))

    monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    monkeypatch.delitem(sys.modules, "sphinx", raising=False)

    with pytest.raises(ImportError, match="requires the 'mlx' library to be installed"):
        importlib.import_module("ml_switcheroo_compiler.backends.mlx")
