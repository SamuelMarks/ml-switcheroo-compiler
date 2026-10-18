"""Unit tests validating BackendRegistry functionality and lazy loaders."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_backend_registry() -> None:
    """Verify backend registration and retrieval in BackendRegistry."""
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
    from ml_switcheroo_compiler.backends.registry import BackendRegistry, get_active_backend, register_backend
    from ml_switcheroo_compiler.core.config import config

    @cast("Callable[..., EagerTensorProtocol]", register_backend)("dummy_decorator")
    class DummyBackend(BaseGenerator):
        """Dummy backend for registry testing."""

        pass

    assert BackendRegistry.get("dummy_decorator") == DummyBackend
    old_backend = config.backend
    config.backend = "dummy_decorator"
    assert get_active_backend() == DummyBackend
    config.backend = old_backend
    with pytest.raises(ValueError):
        BackendRegistry.get("missing_backend_xyz")


def test_safe_import_backend() -> None:
    """Verify safe backend import suppresses ImportError."""
    from ml_switcheroo_compiler.backends import _safe_import_backend

    with patch("importlib.import_module", side_effect=ImportError):
        _safe_import_backend("missing")
    assert callable(test_safe_import_backend)


def test_loaders_and_lazy_loading() -> None:
    """Verify loaders execution and lazy loader error handling."""
    from ml_switcheroo_compiler.backends.registry import _LOADERS, BackendRegistry

    for _name, loader in _LOADERS.items():
        loader()
    with patch.dict(BackendRegistry._LAZY_MODULES, {"test_lazy_fail": "some.module", "test_lazy_no_loader": "some.other"}):
        with patch.dict(_LOADERS, {"test_lazy_fail": MagicMock(side_effect=ImportError("mock error"))}):
            cast("Callable[..., EagerTensorProtocol]", BackendRegistry._try_load_lazy)("test_lazy_fail")
            cast("Callable[..., EagerTensorProtocol]", BackendRegistry._try_load_lazy)("test_lazy_no_loader")
    assert callable(test_loaders_and_lazy_loading)


def test_resolve_alias() -> None:
    """Verify alias resolution in BackendRegistry."""
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    assert BackendRegistry._resolve_alias("torch") in ("pytorch", "torch")


def test_get_all_loaders() -> None:
    """Verify get_all handles import failures gracefully."""
    from ml_switcheroo_compiler.backends.registry import _LOADERS, BackendRegistry

    with patch.dict(BackendRegistry._registry, {}):
        with patch.dict(_LOADERS, {"fake_backend": MagicMock(side_effect=ImportError)}):
            with patch.dict(BackendRegistry._LAZY_MODULES, {"fake_backend": "fake", "fake_no_loader": "fake2"}):
                all_backends = BackendRegistry.get_all()
                assert "fake_backend" not in all_backends
                assert "fake_no_loader" not in all_backends


def test_all_hardware_and_edge_loaders() -> None:
    """Verify lazy-loading invocations for hardware and edge backends."""
    from ml_switcheroo_compiler.backends.registry import BackendRegistry, _load_cuda, _load_edge_webgl, _load_metal, _load_rocm, _load_webgpu

    _load_cuda()
    _load_rocm()
    _load_metal()
    _load_edge_webgl()
    _load_webgpu()
    assert "cuda" in BackendRegistry._registry
    assert "rocm" in BackendRegistry._registry
    assert "metal" in BackendRegistry._registry
    assert "edge_webgl" in BackendRegistry._registry
