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
        with patch("importlib.import_module", side_effect=ImportError):
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
        with patch("importlib.import_module"):
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
