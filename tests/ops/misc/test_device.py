from unittest.mock import patch

from ml_switcheroo_compiler.ops.device import DeviceContextOp, DeviceTransferOp, device_transfer, eval, get_peak_memory, synchronize


def test_device():
    class DummyBackend1:
        pass

    class DummyBackend2:
        def eval(self, *a):
            self.eval_called = True

        def synchronize(self):
            self.sync_called = True

        def get_peak_memory(self):
            return 100

    class DummyArg:
        data = "data"

    arg = DummyArg()

    with patch("ml_switcheroo_compiler.ops.device.get_active_backend", return_value=DummyBackend1(), create=True):
        pass  # Doesn't work, imported inside function

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=DummyBackend1()):
        eval(arg)
        synchronize()
        assert get_peak_memory() == 0

    b2 = DummyBackend2()
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=b2):
        eval(arg)
        assert b2.eval_called
        synchronize()
        assert b2.sync_called
        assert get_peak_memory() == 100

    class DummyTensor:
        shape = (1,)
        dtype = type("D", (), {"value": "float32"})()

    t = DummyTensor()
    assert DeviceContextOp().infer_shape(t) == (1,)
    assert DeviceTransferOp().infer_shape(t) == (1,)

    with patch("ml_switcheroo_compiler.ops.device._emit_shape_node", return_value="emitted"):
        assert device_transfer(t, "cpu") == "emitted"


def test_device_clear_cache_fallback(mocker):
    from ml_switcheroo_compiler.core.device import clear_cache

    mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend", side_effect=Exception("Failed"))
    clear_cache()
