"""Tests for test_device."""

from __future__ import annotations

import typing
from unittest.mock import patch

from ml_switcheroo_compiler.ops.device import DeviceContextOp, DeviceTransferOp, device_transfer, eval, get_peak_memory, synchronize
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_device() -> None:
    """Test function test_device."""

    class DummyBackend1:
        """Test class DummyBackend1."""

        pass

    class DummyBackend2:
        """Test class DummyBackend2."""

        def eval(self, *a: EagerValue) -> None:
            """Test function eval.

            Args:
                a: Test parameter `a`.
            """
            self.eval_called = True

        def synchronize(self) -> None:
            """Test function synchronize."""
            self.sync_called = True

        def get_peak_memory(self) -> EagerValue:
            """Test function get_peak_memory."""
            return 100

    class DummyArg:
        """Test class DummyArg."""

        data = "data"

    arg = DummyArg()

    with patch("ml_switcheroo_compiler.ops.device.get_active_backend", return_value=DummyBackend1(), create=True):
        pass  # Doesn't work, imported inside function

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=DummyBackend1()):
        as_callable(eval)(arg)
        synchronize()
        assert get_peak_memory() == 0

    b2 = DummyBackend2()
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=b2):
        as_callable(eval)(arg)
        assert b2.eval_called
        synchronize()
        assert b2.sync_called
        assert get_peak_memory() == 100

    class DummyTensor:
        """Test class DummyTensor."""

        shape = (1,)
        dtype = type("D", (), {"value": "float32"})()

    t = DummyTensor()
    assert DeviceContextOp().infer_shape(t) == (1,)
    assert DeviceTransferOp().infer_shape(t) == (1,)

    with patch("ml_switcheroo_compiler.ops.device._emit_shape_node", return_value="emitted"):
        assert as_callable(device_transfer)(t, "cpu") == "emitted"


def test_device_clear_cache_fallback(mocker: EagerTensorProtocol) -> None:
    """Test function test_device_clear_cache_fallback.

    Args:
        mocker: Test parameter `mocker`.
    """
    import pytest

    with pytest.raises(Exception, match=r".*"):
        from ml_switcheroo_compiler.core.device import clear_cache

        as_callable(mocker.patch)("ml_switcheroo_compiler.backends.registry.get_active_backend", side_effect=as_callable(Exception)("Failed"))
        clear_cache()
