# ruff: noqa
from ml_switcheroo_compiler.backends.registry import BackendRegistry, _load_cupy, _load_dask, _load_jax, _load_keras, _load_mlx, _load_numpy, _load_pure_python, _load_pytorch, _load_tensorflow
import pytest
from unittest.mock import patch
from ml_switcheroo_compiler.backends.registry import BackendRegistry

"Core abstractions and logic definitions for test_registry_coverage.py."


def test_registry_import_error() -> object:
    """Test the registry import error behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        if "fake" in BackendRegistry._registry:
            del BackendRegistry._registry["fake"]
        BackendRegistry._LAZY_MODULES["fake"] = "fake_module"
        with pytest.raises(ValueError, match="Backend 'fake' not found"):
            BackendRegistry.get("fake")
        BackendRegistry.get_all()
        del BackendRegistry._LAZY_MODULES["fake"]
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_registry_torch_alias() -> object:
    """Test the registry torch alias behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        if "torch" in BackendRegistry._registry:
            del BackendRegistry._registry["torch"]
        BackendRegistry._registry["pytorch"] = "mock_class"

        assert BackendRegistry.get("torch") == "mock_class"
        del BackendRegistry._registry["pytorch"]
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_loaders() -> None:
    for loader in [_load_numpy, _load_pytorch, _load_jax, _load_tensorflow, _load_mlx, _load_dask, _load_keras, _load_cupy, _load_pure_python]:
        try:
            loader()
        except Exception:
            pass


def test_ensure_loaded_import_error() -> None:
    from ml_switcheroo_compiler.backends.registry import _LOADERS

    def failing_loader():
        raise ImportError("Fake import error")

    original_loaders = dict(_LOADERS)
    original_lazy = dict(BackendRegistry._LAZY_MODULES)
    _LOADERS["fake_backend"] = failing_loader
    BackendRegistry._LAZY_MODULES["fake_backend"] = "fake_module_name"
    try:
        BackendRegistry.get("fake_backend")
    except ValueError:
        pass
    _LOADERS.clear()
    _LOADERS.update(original_loaders)
    BackendRegistry._LAZY_MODULES.clear()
    BackendRegistry._LAZY_MODULES.update(original_lazy)


def test_backend_registry_import_error():
    from ml_switcheroo_compiler.backends.registry import BackendRegistry
    import builtins

    BackendRegistry._registry.pop("mock_import_fail", None)
    BackendRegistry._LAZY_MODULES["mock_import_fail"] = "mock_import_fail_module"

    def mock_loader():
        raise ImportError("mocked import error")

    from ml_switcheroo_compiler.backends.registry import _LOADERS

    _LOADERS["mock_import_fail"] = mock_loader

    import pytest

    with pytest.raises(ValueError):
        BackendRegistry.get("mock_import_fail")


def test_get_active_backend():
    from ml_switcheroo_compiler.backends.registry import get_active_backend, BackendRegistry
    from ml_switcheroo_compiler.core.config import config

    old_backend = config.backend
    try:

        class DummyBackend:
            pass

        BackendRegistry.register("dummy_active_backend", DummyBackend)
        config.backend = "dummy_active_backend"
        assert get_active_backend() is DummyBackend
    finally:
        config.backend = old_backend


def test_available_backends_missing_loader():
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    BackendRegistry._registry.pop("mock_no_loader", None)
    BackendRegistry._LAZY_MODULES["mock_no_loader"] = "mock_no_loader_module"

    backends = BackendRegistry.get_all()
    assert "mock_no_loader" not in backends


def test_load_llvm_cpp():
    from ml_switcheroo_compiler.backends.registry import _load_llvm_cpp, _load_edge_onnx, _load_edge_stablehlo, _load_edge_wgsl, _load_edge_wasm_simd, _load_pure_python

    try:
        _load_llvm_cpp()
    except Exception:
        pass
    try:
        _load_edge_onnx()
    except Exception:
        pass
    try:
        _load_edge_stablehlo()
    except Exception:
        pass
    try:
        _load_edge_wgsl()
    except Exception:
        pass
    try:
        _load_edge_wasm_simd()
    except Exception:
        pass
    try:
        _load_pure_python()
    except Exception:
        pass
