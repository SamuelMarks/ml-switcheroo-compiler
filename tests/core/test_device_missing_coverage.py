from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.core.device import Device, DeviceType, Stream, StreamContext, clear_cache, export_function, exporter, get_logical_devices, get_memory_info, get_physical_devices
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


class DummyBackendNoSupport:
    __name__ = "dummy"


def test_device_functions_missing_support():
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=DummyBackendNoSupport()):
        with pytest.raises(BackendNotSupportedError, match="does not support export_function()"):
            export_function()

        with pytest.raises(BackendNotSupportedError, match="does not support get_logical_devices()"):
            get_logical_devices()

        with pytest.raises(BackendNotSupportedError, match="does not support get_physical_devices()"):
            get_physical_devices()

        with pytest.raises(BackendNotSupportedError, match="does not support get_memory_info()"):
            get_memory_info()

        with pytest.warns(UserWarning, match="does not support clear_cache()"):
            clear_cache()


class DummyBackendSupport:
    __name__ = "dummy"

    def export_function(self, *args, **kwargs):
        return True

    def get_logical_devices(self, device_type=None):
        return True

    def get_physical_devices(self, device_type=None):
        return True

    def get_memory_info(self, device=None):
        return True

    def clear_cache(self):
        pass


def test_device_functions_with_support():
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=DummyBackendSupport()):
        export_function()
        assert get_logical_devices()
        assert get_physical_devices()
        assert get_memory_info()
        clear_cache()


def test_function_exporter_enter_exit():
    with exporter(1, 2, a=3) as exp:
        assert exp.args == (1, 2)
        assert exp.kwargs == {"a": 3}
    assert exp.args == ()
    assert exp.kwargs == {}


def test_device_classes():
    d = Device(DeviceType.CPU, 1)
    assert repr(d) == "Device(cpu:1)"

    s = Stream(device=d)
    assert s.device == d

    sc = StreamContext(stream=s)
    with sc as ctx:
        assert ctx.stream == s
    assert sc.stream is None
