"""Test memory_info telemetry across backends."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.backends.dask.generator import DaskGenerator
from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator


def test_pytorch_get_memory_info() -> None:
    """Test PyTorch get_memory_info for both CUDA and CPU fallback paths."""
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.memory_allocated", return_value=1024):
            with patch("torch.cuda.max_memory_allocated", return_value=2048):
                info = PyTorchCodeGenerator.get_memory_info(device="cuda:0")
                assert info["current"] == 1024
                assert info["peak"] == 2048

    with patch("torch.cuda.is_available", return_value=False):
        mock_rusage = MagicMock(ru_maxrss=1000)
        with patch("resource.getrusage", return_value=mock_rusage):
            info = PyTorchCodeGenerator.get_memory_info(device="cpu")
            assert info["current"] == 1000
            assert info["peak"] == 1000


def test_jax_get_memory_info() -> None:
    """Test JAX get_memory_info for both device stats and host fallback paths."""
    mock_stats = {"bytes_in_use": 500, "peak_bytes_in_use": 1500}
    mock_device = MagicMock()
    mock_device.memory_stats.return_value = mock_stats

    with patch("jax.local_devices", return_value=[mock_device]):
        info = JAXCodeGenerator.get_memory_info()
        assert info["current"] == 500
        assert info["peak"] == 1500

    with patch("jax.local_devices", side_effect=RuntimeError("no jax device")):
        mock_rusage = MagicMock(ru_maxrss=500)
        with patch("resource.getrusage", return_value=mock_rusage):
            info = JAXCodeGenerator.get_memory_info()
            assert info["current"] == 500
            assert info["peak"] == 500


def test_dask_get_memory_info() -> None:
    """Test Dask get_memory_info using stdlib resource.getrusage."""
    mock_rusage = MagicMock(ru_maxrss=2000)
    with patch("resource.getrusage", return_value=mock_rusage):
        info = DaskGenerator.get_memory_info()
        assert info["current"] == 2000
        assert info["peak"] == 2000


def test_webgpu_get_memory_info() -> None:
    """Test WebGPU get_memory_info returns zeroed telemetry."""
    info = WebGPUCodeGenerator.get_memory_info()
    assert isinstance(info, dict)
    assert info["current"] == 0
    assert info["peak"] == 0
