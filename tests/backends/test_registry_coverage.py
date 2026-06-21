from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.backends.registry import BackendRegistry


def test_registry_import_error():
    if "fake" in BackendRegistry._registry:
        del BackendRegistry._registry["fake"]

    BackendRegistry._LAZY_MODULES["fake"] = "fake_module"

    with pytest.raises(ValueError, match="Backend 'fake' not found"):
        BackendRegistry.get("fake")

    # Get all with import error
    with patch("importlib.import_module", side_effect=ImportError):
        BackendRegistry.get_all()

    del BackendRegistry._LAZY_MODULES["fake"]


def test_registry_torch_alias():
    if "torch" in BackendRegistry._registry:
        del BackendRegistry._registry["torch"]
    BackendRegistry._registry["pytorch"] = "mock_class"  # type: ignore

    with patch("importlib.import_module"):
        assert BackendRegistry.get("torch") == "mock_class"

    del BackendRegistry._registry["pytorch"]
