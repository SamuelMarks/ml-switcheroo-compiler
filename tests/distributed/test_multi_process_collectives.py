"""Multi-process distributed tests for collective operations: AllReduce, AllGather, ReduceScatter, AllToAll."""

from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np
import pytest
import yaml

from ml_switcheroo_compiler.backends.cuda.nccl_collectives import NCCLDriver
from ml_switcheroo_compiler.distributed.collectives import (
    dispatch_collective,
    get_collective_backend,
    register_collective_backend,
)


def _worker_all_reduce(rank: int, world_size: int, return_dict: dict) -> None:
    """Worker process evaluating simulated multi-rank AllReduce."""
    all_data = [np.array([i + 1.0, (i + 1.0) * 2.0], dtype=np.float32) for i in range(world_size)]
    local_tensor = all_data[rank]
    res = dispatch_collective("AllReduce", local_tensor, backend="numpy", op_type="SUM", rank=rank, all_ranks_data=all_data)
    return_dict[rank] = res


def _worker_all_gather(rank: int, world_size: int, return_dict: dict) -> None:
    """Worker process evaluating simulated multi-rank AllGather."""
    all_data = [np.array([[float(i)]], dtype=np.float32) for i in range(world_size)]
    local_tensor = all_data[rank]
    res = dispatch_collective("AllGather", local_tensor, backend="numpy", axis=0, rank=rank, all_ranks_data=all_data)
    return_dict[rank] = res


def _worker_reduce_scatter(rank: int, world_size: int, return_dict: dict) -> None:
    """Worker process evaluating simulated multi-rank ReduceScatter."""
    # Each rank has 4 elements: [rank, rank*2, rank*3, rank*4]
    all_data = [np.array([float(rank_idx + 1) * (k + 1) for k in range(world_size)], dtype=np.float32) for rank_idx in range(world_size)]
    local_tensor = all_data[rank]
    res = dispatch_collective("ReduceScatter", local_tensor, backend="numpy", op_type="SUM", scatter_dim=0, rank=rank, all_ranks_data=all_data)
    return_dict[rank] = res


def _worker_all_to_all(rank: int, world_size: int, return_dict: dict) -> None:
    """Worker process evaluating simulated multi-rank AllToAll."""
    # Matrix of shape (world_size, 2)
    all_data = [np.arange(world_size * 2, dtype=np.float32).reshape(world_size, 2) + rank_idx * 10 for rank_idx in range(world_size)]
    local_tensor = all_data[rank]
    res = dispatch_collective("AllToAll", local_tensor, backend="numpy", scatter_dim=0, gather_dim=0, rank=rank, all_ranks_data=all_data)
    return_dict[rank] = res


def test_multiprocess_all_reduce() -> None:
    """Test multi-process AllReduce across 4 processes."""
    world_size = 4
    ctx = mp.get_context("fork")
    manager = ctx.Manager()
    return_dict = manager.dict()

    processes = []
    for r in range(world_size):
        p = ctx.Process(target=_worker_all_reduce, args=(r, world_size, return_dict))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Sum of [1, 2], [2, 4], [3, 6], [4, 8] = [10, 20]
    expected = np.array([10.0, 20.0], dtype=np.float32)
    for r in range(world_size):
        np.testing.assert_allclose(return_dict[r], expected)


def test_multiprocess_all_gather() -> None:
    """Test multi-process AllGather across 4 processes."""
    world_size = 4
    ctx = mp.get_context("fork")
    manager = ctx.Manager()
    return_dict = manager.dict()

    processes = []
    for r in range(world_size):
        p = ctx.Process(target=_worker_all_gather, args=(r, world_size, return_dict))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    expected = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    for r in range(world_size):
        np.testing.assert_allclose(return_dict[r], expected)


def test_multiprocess_reduce_scatter() -> None:
    """Test multi-process ReduceScatter across 4 processes."""
    world_size = 4
    ctx = mp.get_context("fork")
    manager = ctx.Manager()
    return_dict = manager.dict()

    processes = []
    for r in range(world_size):
        p = ctx.Process(target=_worker_reduce_scatter, args=(r, world_size, return_dict))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Sum across ranks:
    # rank 0: [1, 2, 3, 4]
    # rank 1: [2, 4, 6, 8]
    # rank 2: [3, 6, 9, 12]
    # rank 3: [4, 8, 12, 16]
    # sum = [10, 20, 30, 40]
    # Chunks: [10], [20], [30], [40]
    for r in range(world_size):
        expected = np.array([(r + 1) * 10.0], dtype=np.float32)
        np.testing.assert_allclose(return_dict[r], expected)


def test_multiprocess_all_to_all() -> None:
    """Test multi-process AllToAll across 4 processes."""
    world_size = 4
    ctx = mp.get_context("fork")
    manager = ctx.Manager()
    return_dict = manager.dict()

    processes = []
    for r in range(world_size):
        p = ctx.Process(target=_worker_all_to_all, args=(r, world_size, return_dict))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for r in range(world_size):
        assert return_dict[r].shape == (4, 2)


def test_backend_collective_dispatchers() -> None:
    """Test backend-specific collective dispatchers."""
    # Custom dispatcher registry
    called = []

    def custom_dispatcher(op: str, tensor: np.ndarray, **kwargs: object) -> np.ndarray:
        called.append(op)
        return tensor * 2

    register_collective_backend("custom_backend", custom_dispatcher)
    assert get_collective_backend("custom_backend") is custom_dispatcher

    inp = np.array([1.0, 2.0], dtype=np.float32)
    res = dispatch_collective("AllReduce", inp, backend="custom_backend")
    assert called == ["AllReduce"]
    np.testing.assert_allclose(res, np.array([2.0, 4.0]))

    # PyTorch and JAX dispatches without initialized cluster
    pt_res = dispatch_collective("AllReduce", inp, backend="pytorch")
    np.testing.assert_allclose(pt_res, inp)
    assert dispatch_collective("AllGather", inp, backend="pytorch") is not None
    assert dispatch_collective("ReduceScatter", inp, backend="pytorch") is not None
    assert dispatch_collective("Broadcast", inp, backend="pytorch") is not None

    jax_res = dispatch_collective("AllReduce", inp, backend="jax")
    np.testing.assert_allclose(jax_res, inp)
    assert dispatch_collective("AllGather", inp, backend="jax") is not None
    assert dispatch_collective("Broadcast", inp, backend="jax") is not None

    # Reduction ops: PROD, MAX, MIN
    all_data = [np.array([2.0, 5.0]), np.array([3.0, 1.0])]
    prod_res = dispatch_collective("AllReduce", all_data[0], backend="numpy", op_type="PROD", all_ranks_data=all_data)
    np.testing.assert_allclose(prod_res, np.array([6.0, 5.0]))

    max_res = dispatch_collective("AllReduce", all_data[0], backend="numpy", op_type="MAX", all_ranks_data=all_data)
    np.testing.assert_allclose(max_res, np.array([3.0, 5.0]))

    min_res = dispatch_collective("AllReduce", all_data[0], backend="numpy", op_type="MIN", all_ranks_data=all_data)
    np.testing.assert_allclose(min_res, np.array([2.0, 1.0]))

    # Broadcast
    bc_res = dispatch_collective("Broadcast", all_data[0], backend="numpy", root=1, all_ranks_data=all_data)
    np.testing.assert_allclose(bc_res, all_data[1])


def test_nccl_driver_bindings() -> None:
    """Test NCCLDriver bindings."""
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    driver = NCCLDriver()
    # Driver safely raises BackendNotSupportedError on non-CUDA host without libnccl
    with pytest.raises(BackendNotSupportedError, match="NCCL library is not available"):
        driver.all_reduce(0, 0, 10)
    with pytest.raises(BackendNotSupportedError, match="NCCL library is not available"):
        driver.all_gather(0, 0, 10)
    with pytest.raises(BackendNotSupportedError, match="NCCL library is not available"):
        driver.broadcast(0, 0, 10)


def test_spmd_sharding_propagation_yaml() -> None:
    """Verify that spmd_mappings/sharding_propagation.yaml is valid."""
    yaml_path = os.path.join(
        os.path.dirname(__file__),
        "../../src/ml_switcheroo_compiler/transforms/passes/spmd_mappings/sharding_propagation.yaml",
    )
    assert os.path.exists(yaml_path)
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    assert "propagation_rules" in data
    assert "matmul" in data["propagation_rules"]
    assert "elementwise" in data["propagation_rules"]
