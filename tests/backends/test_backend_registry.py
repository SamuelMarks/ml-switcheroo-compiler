from unittest.mock import MagicMock, patch

import pytest


def test_backend_registry():
    from ml_switcheroo_compiler.backends.registry import BackendRegistry, get_active_backend, register_backend
    from ml_switcheroo_compiler.core.config import config

    # Test register decorator
    @register_backend("dummy_decorator")
    class DummyBackend:
        pass

    assert BackendRegistry.get("dummy_decorator") == DummyBackend

    # Test active backend
    old_backend = config.backend
    config.backend = "dummy_decorator"
    assert get_active_backend() == DummyBackend
    config.backend = old_backend

    # Test missing backend
    with pytest.raises(ValueError):
        BackendRegistry.get("missing_backend_xyz")


def test_safe_import_backend():
    from ml_switcheroo_compiler.backends import _safe_import_backend

    with patch("importlib.import_module", side_effect=ImportError):
        # Should not raise
        _safe_import_backend("missing")


def test_loaders_and_lazy_loading():
    from ml_switcheroo_compiler.backends.registry import _LOADERS, BackendRegistry

    for name, loader in _LOADERS.items():
        try:
            loader()
        except ImportError:
            pass

    # Test _try_load_lazy for an error condition
    with patch.dict(BackendRegistry._LAZY_MODULES, {"test_lazy_fail": "some.module"}):
        with patch.dict(_LOADERS, {"test_lazy_fail": MagicMock(side_effect=ImportError("mock error"))}):
            # Should not raise, just logs error
            BackendRegistry._try_load_lazy("test_lazy_fail")


def test_resolve_alias():
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    assert BackendRegistry._resolve_alias("torch") in ("pytorch", "torch")


def test_get_all_loaders():
    from ml_switcheroo_compiler.backends.registry import _LOADERS, BackendRegistry

    with patch.dict(BackendRegistry._registry, {}):
        with patch.dict(_LOADERS, {"fake_backend": MagicMock(side_effect=ImportError)}):
            with patch.dict(BackendRegistry._LAZY_MODULES, {"fake_backend": "fake"}):
                # get_all should suppress ImportError
                all_backends = BackendRegistry.get_all()
                assert "fake_backend" not in all_backends
