"""Test core device functions extended."""

from unittest.mock import MagicMock

from ml_switcheroo_compiler.core.device import (
    export_function,
    get_logical_devices,
    get_memory_info,
    get_physical_devices,
)


def test_export_function_call(monkeypatch):
    mock_backend = MagicMock()
    mock_backend.export_function = MagicMock()

    import ml_switcheroo_compiler.backends.registry as registry_module

    monkeypatch.setattr(registry_module, "get_active_backend", lambda: mock_backend)

    export_function("arg1", kwarg1="val1")
    mock_backend.export_function.assert_called_once_with("arg1", kwarg1="val1")

    # Test exception block
    mock_backend.export_function.side_effect = Exception("test error")
    export_function()  # Should not raise


def test_get_logical_devices_success(monkeypatch):
    mock_backend = MagicMock()
    mock_backend.get_logical_devices.return_value = ["mock_dev"]

    import ml_switcheroo_compiler.backends.registry as registry_module

    monkeypatch.setattr(registry_module, "get_active_backend", lambda: mock_backend)

    assert get_logical_devices("gpu") == ["mock_dev"]

    # Test exception block
    mock_backend.get_logical_devices.side_effect = Exception("test error")
    assert get_logical_devices("gpu")[0].device_type.value == "gpu"


def test_get_physical_devices_success(monkeypatch):
    mock_backend = MagicMock()
    mock_backend.get_physical_devices.return_value = ["mock_dev_p"]

    import ml_switcheroo_compiler.backends.registry as registry_module

    monkeypatch.setattr(registry_module, "get_active_backend", lambda: mock_backend)

    assert get_physical_devices("gpu") == ["mock_dev_p"]

    # Test exception block
    mock_backend.get_physical_devices.side_effect = Exception("test error")
    assert get_physical_devices("gpu")[0].device_type.value == "gpu"


def test_get_memory_info_success(monkeypatch):
    mock_backend = MagicMock()
    mock_backend.get_memory_info.return_value = {"current": 100, "peak": 200}

    import ml_switcheroo_compiler.backends.registry as registry_module

    monkeypatch.setattr(registry_module, "get_active_backend", lambda: mock_backend)

    assert get_memory_info("gpu") == {"current": 100, "peak": 200}

    # Test exception block
    mock_backend.get_memory_info.side_effect = Exception("test error")
    assert get_memory_info("gpu") == {"current": 0, "peak": 0}
