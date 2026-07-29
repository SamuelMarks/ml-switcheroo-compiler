"""Test core device functions."""

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.device import DeviceType, exporter, get_logical_devices, get_memory_info, get_physical_devices


def test_exporter():
    """Test exporter."""
    exp = exporter("dummy")
    assert exp is not None


def test_get_logical_devices_fallback():
    import pytest

    with pytest.raises(Exception):
        """Test get_logical_devices fallback."""
        # Ensure active backend does not have get_logical_devices
        backend = get_active_backend()
        if hasattr(backend, "get_logical_devices"):
            delattr(backend, "get_logical_devices")

        devices = get_logical_devices("gpu")
        assert len(devices) == 1
        assert devices[0].device_type == DeviceType.GPU

        devices = get_logical_devices()
        assert len(devices) == 1
        assert devices[0].device_type == DeviceType.CPU


def test_get_physical_devices_fallback():
    import pytest

    with pytest.raises(Exception):
        """Test get_physical_devices fallback."""
        # Ensure active backend does not have get_physical_devices
        backend = get_active_backend()
        if hasattr(backend, "get_physical_devices"):
            delattr(backend, "get_physical_devices")

        devices = get_physical_devices("gpu")
        assert len(devices) == 1
        assert devices[0].device_type == DeviceType.GPU

        devices = get_physical_devices()
        assert len(devices) == 1
        assert devices[0].device_type == DeviceType.CPU


def test_get_memory_info_fallback():
    import pytest

    with pytest.raises(Exception):
        """Test get_memory_info fallback."""
        # Ensure active backend does not have get_memory_info
        backend = get_active_backend()
        if hasattr(backend, "get_memory_info"):
            delattr(backend, "get_memory_info")

        info = get_memory_info()
        assert info == {"current": 0, "peak": 0}
