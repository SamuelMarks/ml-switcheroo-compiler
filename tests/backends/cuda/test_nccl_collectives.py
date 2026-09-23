"""Tests for NCCL collective driver bindings."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.cuda.nccl_collectives import (
    NCCL_FLOAT32,
    NCCL_FLOAT64,
    NCCL_INT32,
    NCCL_INT64,
    NCCL_MAX,
    NCCL_MIN,
    NCCL_PROD,
    NCCL_SUM,
    NCCLDriver,
)
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.distributed.host_collectives import HostCollectiveCommunicator


def test_nccl_constants() -> None:
    """Verify NCCL constants."""
    assert NCCL_FLOAT32 == 7
    assert NCCL_FLOAT64 == 8
    assert NCCL_INT32 == 2
    assert NCCL_INT64 == 3
    assert NCCL_SUM == 0
    assert NCCL_PROD == 1
    assert NCCL_MAX == 2
    assert NCCL_MIN == 3


def test_nccl_driver_init_empty_path() -> None:
    """Verify NCCLDriver handles empty lib_path."""
    with patch("ctypes.CDLL", side_effect=OSError("Not found")):
        driver = NCCLDriver(lib_path="")
        assert not driver.is_available()
        assert driver.nccl_lib is None


def test_nccl_driver_init_success() -> None:
    """Verify NCCLDriver initializes successfully when CDLL succeeds."""
    mock_lib = MagicMock()
    with patch("ctypes.CDLL", return_value=mock_lib):
        driver = NCCLDriver(lib_path="/path/to/libnccl.so")
        assert driver.is_available()
        assert driver.nccl_lib is mock_lib


def test_nccl_driver_unavailable_branches() -> None:
    """Verify all collectives raise BackendNotSupportedError when driver is unavailable or nccl_lib is None."""
    driver = NCCLDriver()
    driver.nccl_lib = None
    with pytest.raises(BackendNotSupportedError, match="NCCL library is not available"):
        driver.all_reduce(1, 2, 10)
    with pytest.raises(BackendNotSupportedError, match="NCCL library is not available"):
        driver.all_gather(1, 2, 10)
    with pytest.raises(BackendNotSupportedError, match="NCCL library is not available"):
        driver.broadcast(1, 2, 10)
    with pytest.raises(BackendNotSupportedError, match="NCCL library is not available"):
        driver.reduce_scatter(1, 2, 10)
    with pytest.raises(BackendNotSupportedError, match="NCCL library is not available"):
        driver.all_to_all(1, 2, 10)
    with pytest.raises(BackendNotSupportedError, match="NCCL library is not available"):
        driver.send(1, 10)
    with pytest.raises(BackendNotSupportedError, match="NCCL library is not available"):
        driver.recv(2, 10)
    with pytest.raises(BackendNotSupportedError, match="NCCL library is not available"):
        driver.get_unique_id()
    with pytest.raises(BackendNotSupportedError, match="NCCL library is not available"):
        driver.init_rank(2, b"x" * 128, 0)
    with pytest.raises(BackendNotSupportedError, match="NCCL library is not available"):
        driver.comm_destroy(123)
    with pytest.raises(BackendNotSupportedError, match="NCCL library is not available"):
        driver.group_start()
    with pytest.raises(BackendNotSupportedError, match="NCCL library is not available"):
        driver.group_end()


def test_nccl_driver_all_reduce() -> None:
    """Verify all_reduce success and error paths."""
    mock_lib = MagicMock()
    mock_lib.ncclAllReduce.return_value = 0

    with patch("ctypes.CDLL", return_value=mock_lib):
        driver = NCCLDriver()

        # Success with default comm and stream (falsy branch: None -> 0)
        res1 = driver.all_reduce(100, 200, 4, datatype=NCCL_FLOAT32, op=NCCL_SUM, comm=None, stream=None)
        assert res1 == 0
        assert mock_lib.ncclAllReduce.called

        # Success with explicit comm and stream (truthy branch)
        res2 = driver.all_reduce(100, 200, 4, datatype=NCCL_FLOAT64, op=NCCL_MAX, comm=123, stream=456)
        assert res2 == 0

        # Exception path
        mock_lib.ncclAllReduce.side_effect = RuntimeError("Reduction failure")
        res_err = driver.all_reduce(100, 200, 4)
        assert res_err == -1


def test_nccl_driver_all_gather() -> None:
    """Verify all_gather success and error paths."""
    mock_lib = MagicMock()
    mock_lib.ncclAllGather.return_value = 0

    with patch("ctypes.CDLL", return_value=mock_lib):
        driver = NCCLDriver()

        # Success with default comm and stream (falsy branch: None -> 0)
        res1 = driver.all_gather(100, 200, 4, datatype=NCCL_INT32, comm=None, stream=None)
        assert res1 == 0
        assert mock_lib.ncclAllGather.called

        # Success with explicit comm and stream (truthy branch)
        res2 = driver.all_gather(100, 200, 4, datatype=NCCL_INT64, comm=789, stream=101)
        assert res2 == 0

        # Exception path
        mock_lib.ncclAllGather.side_effect = RuntimeError("Gather failure")
        res_err = driver.all_gather(100, 200, 4)
        assert res_err == -1


def test_nccl_driver_broadcast() -> None:
    """Verify broadcast success and error paths."""
    mock_lib = MagicMock()
    mock_lib.ncclBroadcast.return_value = 0

    with patch("ctypes.CDLL", return_value=mock_lib):
        driver = NCCLDriver()

        # Success with default comm and stream (falsy branch: None -> 0)
        res1 = driver.broadcast(100, 200, 4, datatype=NCCL_FLOAT32, root=0, comm=None, stream=None)
        assert res1 == 0
        assert mock_lib.ncclBroadcast.called

        # Success with explicit comm and stream (truthy branch)
        res2 = driver.broadcast(100, 200, 4, datatype=NCCL_FLOAT32, root=1, comm=321, stream=654)
        assert res2 == 0

        # Exception path
        mock_lib.ncclBroadcast.side_effect = RuntimeError("Broadcast failure")
        res_err = driver.broadcast(100, 200, 4)
        assert res_err == -1


def test_nccl_driver_reduce_scatter() -> None:
    """Verify reduce_scatter success and error paths."""
    mock_lib = MagicMock()
    mock_lib.ncclReduceScatter.return_value = 0

    with patch("ctypes.CDLL", return_value=mock_lib):
        driver = NCCLDriver()

        # Success with default comm and stream
        res1 = driver.reduce_scatter(100, 200, 4, datatype=NCCL_FLOAT32, op=NCCL_SUM, comm=None, stream=None)
        assert res1 == 0
        assert mock_lib.ncclReduceScatter.called

        # Success with explicit comm and stream
        res2 = driver.reduce_scatter(100, 200, 4, datatype=NCCL_FLOAT64, op=NCCL_PROD, comm=55, stream=66)
        assert res2 == 0

        # Exception path
        mock_lib.ncclReduceScatter.side_effect = RuntimeError("ReduceScatter failure")
        res_err = driver.reduce_scatter(100, 200, 4)
        assert res_err == -1


def test_nccl_driver_send_and_recv() -> None:
    """Verify point-to-point send and recv bindings."""
    mock_lib = MagicMock()
    mock_lib.ncclSend.return_value = 0
    mock_lib.ncclRecv.return_value = 0

    with patch("ctypes.CDLL", return_value=mock_lib):
        driver = NCCLDriver()

        # Send success with defaults
        res_s1 = driver.send(100, 4, datatype=NCCL_FLOAT32, peer=1, comm=None, stream=None)
        assert res_s1 == 0
        assert mock_lib.ncclSend.called

        # Send with explicit comm and stream
        res_s2 = driver.send(100, 4, datatype=NCCL_FLOAT32, peer=2, comm=11, stream=22)
        assert res_s2 == 0

        # Recv success with defaults
        res_r1 = driver.recv(200, 4, datatype=NCCL_FLOAT32, peer=1, comm=None, stream=None)
        assert res_r1 == 0
        assert mock_lib.ncclRecv.called

        # Recv with explicit comm and stream
        res_r2 = driver.recv(200, 4, datatype=NCCL_FLOAT32, peer=0, comm=33, stream=44)
        assert res_r2 == 0

        # Exception paths
        mock_lib.ncclSend.side_effect = RuntimeError("Send failure")
        assert driver.send(100, 4) == -1

        mock_lib.ncclRecv.side_effect = RuntimeError("Recv failure")
        assert driver.recv(200, 4) == -1


def test_nccl_driver_dll_fallback() -> None:
    """Verify driver successfully discovers and falls back to nccl.dll."""
    mock_lib = MagicMock()

    def mock_cdll(path: str) -> MagicMock:
        if path == "nccl.dll":
            return mock_lib
        raise OSError("Library not found")

    with patch("ctypes.CDLL", side_effect=mock_cdll):
        driver = NCCLDriver()
        assert driver.is_available()
        assert driver.nccl_lib is mock_lib


def test_nccl_driver_all_to_all_and_lifecycle() -> None:
    """Verify all_to_all, get_unique_id, init_rank, and comm_destroy."""
    mock_lib = MagicMock()
    mock_lib.ncclAllToAll.return_value = 0
    mock_lib.ncclGetUniqueId.return_value = 0
    mock_lib.ncclCommInitRank.return_value = 0
    mock_lib.ncclCommDestroy.return_value = 0
    mock_lib.ncclGroupStart.return_value = 0
    mock_lib.ncclGroupEnd.return_value = 0

    with patch("ctypes.CDLL", return_value=mock_lib):
        driver = NCCLDriver()

        # all_to_all
        assert driver.all_to_all(100, 200, 4) == 0

        # unique_id
        uid = driver.get_unique_id()
        assert isinstance(uid, bytes)

        # init_rank & destroy
        comm = driver.init_rank(2, uid, 0)
        assert comm == 0
        assert driver.comm_destroy(comm) == 0

        # group calls & stream sync
        assert driver.group_start() == 0
        assert driver.group_end() == 0
        assert driver.stream_synchronize(None) == 0


def test_host_collective_communicator() -> None:
    """Verify HostCollectiveCommunicator operations for CPU fallback."""
    r0 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    r1 = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    all_data = [r0, r1]

    comm0 = HostCollectiveCommunicator(world_size=2, rank=0)
    comm1 = HostCollectiveCommunicator(world_size=2, rank=1)

    # AllReduce SUM
    red_sum = comm0.all_reduce(r0, op="SUM", all_ranks_data=all_data)
    np.testing.assert_allclose(red_sum, [6.0, 8.0, 10.0, 12.0])

    # AllReduce MAX
    red_max = comm1.all_reduce(r1, op="MAX", all_ranks_data=all_data)
    np.testing.assert_allclose(red_max, [5.0, 6.0, 7.0, 8.0])

    # AllGather
    gathered = comm0.all_gather(r0, axis=0, all_ranks_data=all_data)
    assert len(gathered) == 8

    # ReduceScatter
    rs0 = comm0.reduce_scatter(r0, op="SUM", scatter_dim=0, all_ranks_data=all_data)
    rs1 = comm1.reduce_scatter(r1, op="SUM", scatter_dim=0, all_ranks_data=all_data)
    np.testing.assert_allclose(rs0, [6.0, 8.0])
    np.testing.assert_allclose(rs1, [10.0, 12.0])

    # AllToAll
    a2a0 = comm0.all_to_all(r0, scatter_dim=0, gather_dim=0, all_ranks_data=all_data)
    assert len(a2a0) == 4

    # Broadcast
    bc = comm1.broadcast(r1, root=0, all_ranks_data=all_data)
    np.testing.assert_allclose(bc, r0)

    # Additional branches: MIN, PROD, unknown op
    red_min = comm0.all_reduce(r0, op="MIN", all_ranks_data=all_data)
    np.testing.assert_allclose(red_min, [1.0, 2.0, 3.0, 4.0])
    red_prod = comm0.all_reduce(r0, op="PROD", all_ranks_data=all_data)
    np.testing.assert_allclose(red_prod, [5.0, 12.0, 21.0, 32.0])
    red_other = comm0.all_reduce(r0, op="CUSTOM", all_ranks_data=all_data)
    np.testing.assert_allclose(red_other, [6.0, 8.0, 10.0, 12.0])

    # Branches where all_ranks_data is None
    np.testing.assert_allclose(comm0.all_reduce(r0, all_ranks_data=None), r0)
    np.testing.assert_allclose(comm0.all_gather(r0, all_ranks_data=None), r0)
    np.testing.assert_allclose(comm0.all_to_all(r0, all_ranks_data=None), r0)
    np.testing.assert_allclose(comm0.broadcast(r0, all_ranks_data=None), r0)
    np.testing.assert_allclose(comm0.reduce_scatter(r0, all_ranks_data=None), [1.0, 2.0])


def test_nccl_driver_exhaustive_branches() -> None:
    """Verify all remaining exception and non-zero status branches in NCCLDriver."""
    mock_lib = MagicMock()

    with patch("ctypes.CDLL", return_value=mock_lib):
        driver = NCCLDriver()

        # all_to_all without hasattr(ncclAllToAll)
        mock_no_a2a = MagicMock(spec=[])
        driver.nccl_lib = mock_no_a2a
        assert driver.all_to_all(100, 200, 4) == 0

        # all_to_all exception path
        mock_a2a_err = MagicMock()
        mock_a2a_err.ncclAllToAll.side_effect = RuntimeError("AllToAll error")
        driver.nccl_lib = mock_a2a_err
        assert driver.all_to_all(100, 200, 4) == -1

        # get_unique_id failure status != 0
        mock_uid_err = MagicMock()
        mock_uid_err.ncclGetUniqueId.return_value = 1
        driver.nccl_lib = mock_uid_err
        with pytest.raises(RuntimeError, match="ncclGetUniqueId failed with status 1"):
            driver.get_unique_id()

        # init_rank failure status != 0
        mock_init_err = MagicMock()
        mock_init_err.ncclCommInitRank.return_value = 2
        driver.nccl_lib = mock_init_err
        with pytest.raises(RuntimeError, match="ncclCommInitRank failed with status 2"):
            driver.init_rank(2, b"x" * 128, 0)

        # comm_destroy exception path
        mock_destroy_err = MagicMock()
        mock_destroy_err.ncclCommDestroy.side_effect = RuntimeError("Destroy error")
        driver.nccl_lib = mock_destroy_err
        assert driver.comm_destroy(123) == -1

        # group_start exception path
        mock_start_err = MagicMock()
        mock_start_err.ncclGroupStart.side_effect = RuntimeError("GroupStart error")
        driver.nccl_lib = mock_start_err
        assert driver.group_start() == -1

        # group_end exception path
        mock_end_err = MagicMock()
        mock_end_err.ncclGroupEnd.side_effect = RuntimeError("GroupEnd error")
        driver.nccl_lib = mock_end_err
        assert driver.group_end() == -1


def test_nccl_driver_cuda_rt_memory_and_streams() -> None:
    """Verify cuda_rt library discovery, cudaMalloc, cudaFree, cudaMemcpy, and stream_synchronize."""
    import ctypes

    mock_rt = MagicMock()
    mock_nccl = MagicMock()

    # Successful discovery of both cuda_rt and nccl
    with patch("ctypes.CDLL", side_effect=[mock_rt, mock_nccl]):
        driver = NCCLDriver()
        assert driver.cuda_rt is mock_rt
        assert driver.nccl_lib is mock_nccl

    # cudaMalloc success path
    def fake_malloc_success(ptr_ref: object, size: int) -> int:
        """Simulate successful cudaMalloc.

        Args:
            ptr_ref (object): Pointer reference.
            size (int): Allocation size.

        Returns:
            int: Return status code.
        """
        ctypes.cast(ptr_ref, ctypes.POINTER(ctypes.c_void_p)).contents.value = 0x12345678
        return 0

    mock_rt.cudaMalloc = fake_malloc_success
    driver.cuda_rt = mock_rt
    dev_ptr = driver.allocate_buffer(128)
    assert dev_ptr == 0x12345678
    assert dev_ptr not in driver._allocated_buffers

    # cudaFree on hardware device pointer
    mock_rt.cudaFree.return_value = 0
    driver.free_buffer(dev_ptr)
    assert mock_rt.cudaFree.called

    # cudaFree exception handling
    mock_rt.cudaFree.side_effect = RuntimeError("cudaFree failed")
    driver.free_buffer(dev_ptr)  # Should not raise

    # cudaMemcpy H2D success and exception
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mock_rt.cudaMemcpy.return_value = 0
    mock_rt.cudaMemcpy.side_effect = None
    driver.copy_host_to_device(arr, dev_ptr)
    assert mock_rt.cudaMemcpy.called

    # cudaMemcpy H2D exception fallback to ctypes.memmove (with real allocated buffer)
    del mock_rt.cudaMalloc
    host_buf_ptr = driver.allocate_buffer(arr.nbytes)
    real_buf = driver._allocated_buffers.pop(host_buf_ptr)
    mock_rt.cudaMemcpy.side_effect = RuntimeError("H2D failed")
    driver.copy_host_to_device(arr, host_buf_ptr)

    # cudaMemcpy D2H success and exception
    out_arr = np.zeros_like(arr)
    mock_rt.cudaMemcpy.side_effect = None
    driver.copy_device_to_host(dev_ptr, out_arr)
    assert mock_rt.cudaMemcpy.called

    mock_rt.cudaMemcpy.side_effect = RuntimeError("D2H failed")
    driver.copy_device_to_host(host_buf_ptr, out_arr)
    assert np.allclose(out_arr, arr)
    del real_buf

    # Restore cudaMalloc with non-zero status branch (fails cudaMalloc, falls back to host buffer)
    mock_rt.cudaMalloc = MagicMock(return_value=1)
    fallback_buf = driver.allocate_buffer(64)
    assert fallback_buf in driver._allocated_buffers

    # stream_synchronize success, None, and exception
    mock_rt.cudaStreamSynchronize.side_effect = None
    mock_rt.cudaStreamSynchronize.return_value = 0
    assert driver.stream_synchronize(None) == 0
    assert driver.stream_synchronize(999) == 0

    mock_rt.cudaStreamSynchronize.side_effect = RuntimeError("Sync fail")
    assert driver.stream_synchronize(999) == -1

    # cuda_rt without stream synchronize
    mock_no_sync = MagicMock(spec=[])
    driver.cuda_rt = mock_no_sync
    assert driver.stream_synchronize(999) == 0
