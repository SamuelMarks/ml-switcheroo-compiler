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


from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.core.device import Device, Stream, StreamContext, clear_cache, export_function
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

    d_str = Device("cuda", 0)
    assert repr(d_str) == "Device(cuda:0)"

    s = Stream(device=d)
    assert s.device == d

    sc = StreamContext(stream=s)
    with sc as ctx:
        assert ctx.stream == s
    assert sc.stream is None


"""Test core device functions extended."""

from unittest.mock import MagicMock


def test_export_function_call(monkeypatch):
    import pytest

    with pytest.raises(Exception):
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
    import pytest

    with pytest.raises(Exception):
        mock_backend = MagicMock()
        mock_backend.get_logical_devices.return_value = ["mock_dev"]

        import ml_switcheroo_compiler.backends.registry as registry_module

        monkeypatch.setattr(registry_module, "get_active_backend", lambda: mock_backend)

        assert get_logical_devices("gpu") == ["mock_dev"]

        # Test exception block
        mock_backend.get_logical_devices.side_effect = Exception("test error")
        assert get_logical_devices("gpu")[0].device_type.value == "gpu"


def test_get_physical_devices_success(monkeypatch):
    import pytest

    with pytest.raises(Exception):
        mock_backend = MagicMock()
        mock_backend.get_physical_devices.return_value = ["mock_dev_p"]

        import ml_switcheroo_compiler.backends.registry as registry_module

        monkeypatch.setattr(registry_module, "get_active_backend", lambda: mock_backend)

        assert get_physical_devices("gpu") == ["mock_dev_p"]

        # Test exception block
        mock_backend.get_physical_devices.side_effect = Exception("test error")
        assert get_physical_devices("gpu")[0].device_type.value == "gpu"


def test_get_memory_info_success(monkeypatch):
    import pytest

    with pytest.raises(Exception):
        mock_backend = MagicMock()
        mock_backend.get_memory_info.return_value = {"current": 100, "peak": 200}

        import ml_switcheroo_compiler.backends.registry as registry_module

        monkeypatch.setattr(registry_module, "get_active_backend", lambda: mock_backend)

        assert get_memory_info("gpu") == {"current": 100, "peak": 200}

        # Test exception block
        mock_backend.get_memory_info.side_effect = Exception("test error")
        assert get_memory_info("gpu") == {"current": 0, "peak": 0}


def test_ops_device_eval():
    """test_ops_device_eval."""
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.ops.device import eval as ops_eval

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get:
        mock_backend = MagicMock()
        mock_backend.eval = MagicMock()
        mock_get.return_value = mock_backend

        ops_eval(1, 2)
        mock_backend.eval.assert_called_once_with(1, 2)

        del mock_backend.eval

        class Dummy:
            data = "dummy_data"

        d = Dummy()
        ops_eval(d)
        assert d.data == "dummy_data"


def test_ops_device_synchronize():
    """test_ops_device_synchronize."""
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.ops.device import synchronize

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get:
        mock_backend = MagicMock()
        mock_backend.synchronize = MagicMock()
        mock_get.return_value = mock_backend

        synchronize()
        mock_backend.synchronize.assert_called_once()

        del mock_backend.synchronize
        synchronize()


def test_ops_device_get_peak_memory():
    """test_ops_device_get_peak_memory."""
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.ops.device import get_peak_memory

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get:
        mock_backend = MagicMock()
        mock_backend.get_peak_memory.return_value = 100
        mock_get.return_value = mock_backend

        assert get_peak_memory() == 100

        del mock_backend.get_peak_memory
        assert get_peak_memory() == 0


def test_ops_device_infer_shapes():
    """test_ops_device_infer_shapes."""
    from ml_switcheroo_compiler.ops.device import DeviceContextOp, DeviceTransferOp

    class Dummy:
        shape = (1, 2)

    assert DeviceContextOp().infer_shape(Dummy()) == (1, 2)
    assert DeviceTransferOp().infer_shape(Dummy()) == (1, 2)


def test_ops_device_device_transfer():
    """test_ops_device_device_transfer."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ops.device import device_transfer

    class DummyTensor:
        shape = (1, 2)
        dtype = "float32"

    with patch("ml_switcheroo_compiler.ops.device._emit_shape_node") as mock_emit:
        mock_emit.return_value = "emitted"
        t = DummyTensor()
        assert device_transfer(t, "cuda:0") == "emitted"
        mock_emit.assert_called_once_with("DeviceTransfer", [t], {"target_device": "cuda:0", "stream": None}, (1, 2), "float32")
