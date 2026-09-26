"""Test core device functions and unified backend device interface."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo_compiler.backends.cuda.cuda import CudaCodeGenerator
from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.backends.metal.metal import MetalCodeGenerator
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.backends.rocm.rocm import RocmCodeGenerator
from ml_switcheroo_compiler.core.device import (
    Device,
    DeviceType,
    Stream,
    StreamContext,
    clear_cache,
    export_function,
    exporter,
    get_logical_devices,
    get_memory_info,
    get_physical_devices,
)
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.ops.device import DeviceContextOp, DeviceTransferOp, device_transfer, get_peak_memory, synchronize
from ml_switcheroo_compiler.ops.device import eval as ops_eval


def test_exporter() -> None:
    """Test exporter context manager."""
    exp = exporter("dummy")
    assert exp is not None
    with exporter(1, 2, a=3) as ctx:
        assert ctx.args == (1, 2)
        assert ctx.kwargs == {"a": 3}
    assert ctx.args == ()
    assert ctx.kwargs == {}


def test_real_backend_device_queries() -> None:
    """Verify device and memory info queries directly against the real registered active backend."""
    backend = get_active_backend()
    assert backend is not None

    # Real logical device query
    devices = get_logical_devices()
    assert len(devices) >= 1
    assert any(d.device_type == DeviceType.CPU for d in devices)

    devices_cpu = get_logical_devices("cpu")
    assert len(devices_cpu) >= 1
    assert all(d.device_type == DeviceType.CPU for d in devices_cpu)

    # Real physical device query
    physical = get_physical_devices()
    assert len(physical) >= 1
    assert any(d.device_type == DeviceType.CPU for d in physical)

    # Real memory info query
    mem_info = get_memory_info()
    assert "current" in mem_info
    assert "peak" in mem_info

    # Real export_function invocation
    export_function("model_trace")


def test_hardware_backend_device_interfaces() -> None:
    """Verify hardware accelerator backends report appropriate GPU/WebGPU devices."""
    # CUDA
    cuda_devs = CudaCodeGenerator.get_logical_devices()
    assert any(d.device_type == DeviceType.GPU for d in cuda_devs)
    assert len(CudaCodeGenerator.get_logical_devices("gpu")) >= 1
    assert len(CudaCodeGenerator.get_logical_devices("cuda")) >= 1
    assert len(CudaCodeGenerator.get_logical_devices("cpu")) >= 1
    assert CudaCodeGenerator.get_physical_devices() is not None
    assert CudaCodeGenerator.get_memory_info() == {"current": 0, "peak": 0}
    CudaCodeGenerator.initialize_distributed()
    CudaCodeGenerator.export_function()

    # ROCm
    rocm_devs = RocmCodeGenerator.get_logical_devices()
    assert any(d.device_type == DeviceType.GPU for d in rocm_devs)
    assert len(RocmCodeGenerator.get_logical_devices("rocm")) >= 1

    # Metal
    metal_devs = MetalCodeGenerator.get_logical_devices()
    assert any(d.device_type == DeviceType.GPU for d in metal_devs)
    assert len(MetalCodeGenerator.get_logical_devices("metal")) >= 1

    # WebGPU
    webgpu_devs = WebGPUCodeGenerator.get_logical_devices()
    assert any(d.device_type == DeviceType.WEBGPU for d in webgpu_devs)
    assert len(WebGPUCodeGenerator.get_logical_devices("webgpu")) >= 1


class DummyBackendNoSupport:
    """Dummy backend lacking optional device interfaces."""

    __name__ = "dummy"


def test_device_functions_missing_support() -> None:
    """Verify informative errors when active backend lacks device interfaces."""
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

    dummy_with_cache = MagicMock()
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=dummy_with_cache):
        clear_cache()
        dummy_with_cache.clear_cache.assert_called_once()


def test_device_classes() -> None:
    """Test Device, Stream, and StreamContext structures."""
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


def test_ops_device_eval() -> None:
    """Test ops.device eval helper."""
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


def test_ops_device_synchronize() -> None:
    """Test ops.device synchronize helper."""
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get:
        mock_backend = MagicMock()
        mock_backend.synchronize = MagicMock()
        mock_get.return_value = mock_backend

        synchronize()
        mock_backend.synchronize.assert_called_once()

        del mock_backend.synchronize
        synchronize()


def test_ops_device_get_peak_memory() -> None:
    """Test ops.device get_peak_memory helper."""
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get:
        mock_backend = MagicMock()
        mock_backend.get_peak_memory.return_value = 100
        mock_get.return_value = mock_backend

        assert get_peak_memory() == 100

        del mock_backend.get_peak_memory
        assert get_peak_memory() == 0


def test_ops_device_infer_shapes() -> None:
    """Test ops.device infer_shape methods."""

    class Dummy:
        shape = (1, 2)

    assert DeviceContextOp().infer_shape(Dummy()) == (1, 2)
    assert DeviceTransferOp().infer_shape(Dummy()) == (1, 2)


def test_ops_device_device_transfer() -> None:
    """Test ops.device device_transfer dispatch."""

    class DummyTensor:
        shape = (1, 2)
        dtype = "float32"

    with patch("ml_switcheroo_compiler.ops.device._emit_shape_node") as mock_emit:
        mock_emit.return_value = "emitted"
        t = DummyTensor()
        assert device_transfer(t, "cuda:0") == "emitted"
        mock_emit.assert_called_once_with(
            "DeviceTransfer",
            [t],
            {"target_device": "cuda:0", "stream": None},
            (1, 2),
            "float32",
        )
