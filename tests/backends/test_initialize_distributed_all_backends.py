"""Test initialize_distributed across all backends."""

from __future__ import annotations

from unittest.mock import patch

from ml_switcheroo_compiler.backends.dask.generator import DaskGenerator
from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
from ml_switcheroo_compiler.distributed.distributed_config import DistributedConfig


def test_pytorch_initialize_distributed() -> None:
    """Test PyTorch initialize_distributed."""
    with patch("torch.distributed.is_initialized", return_value=False):
        with patch("torch.distributed.init_process_group") as mock_init:
            config = DistributedConfig(backend="nccl", init_method="env://", world_size=4, rank=0)
            PyTorchCodeGenerator.initialize_distributed(backend_config=config)
            mock_init.assert_called_once_with(backend="nccl", init_method="env://", world_size=4, rank=0)

    with patch("torch.distributed.is_initialized", return_value=True):
        with patch("torch.distributed.init_process_group") as mock_init:
            PyTorchCodeGenerator.initialize_distributed()
            mock_init.assert_not_called()


def test_jax_initialize_distributed() -> None:
    """Test JAX initialize_distributed."""
    with patch("jax.distributed.initialize") as mock_init:
        config = DistributedConfig(world_size=4, rank=0, options={"coordinator_address": "localhost:1234"})
        JAXCodeGenerator.initialize_distributed(backend_config=config)
        mock_init.assert_called_once_with(coordinator_address="localhost:1234", num_processes=4, process_id=0)

    with patch("jax.distributed.initialize") as mock_init:
        JAXCodeGenerator.initialize_distributed()
        mock_init.assert_called_once_with()


def test_dask_initialize_distributed() -> None:
    """Test Dask initialize_distributed."""
    with patch("dask.distributed.Client") as mock_client:
        config = DistributedConfig(options={"scheduler_address": "localhost:8786"})
        DaskGenerator.initialize_distributed(backend_config=config)
        mock_client.assert_called_once_with("localhost:8786")

    with patch("dask.distributed.Client") as mock_client:
        DaskGenerator.initialize_distributed()
        mock_client.assert_called_once_with()


def test_webgpu_initialize_distributed() -> None:
    """Test WebGPU initialize_distributed."""
    # WebGPU initialization is a no-op on the python side.
    config = DistributedConfig()
    WebGPUCodeGenerator.initialize_distributed(backend_config=config)
    assert config is not None
