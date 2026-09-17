"""Tests for RCCL collective driver bindings for ROCm/HIP."""

from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo_compiler.backends.rocm.rccl_collectives import (
    RCCL_FLOAT32,
    RCCL_FLOAT64,
    RCCL_INT32,
    RCCL_INT64,
    RCCL_MAX,
    RCCL_MIN,
    RCCL_PROD,
    RCCL_SUM,
    RCCLDriver,
)
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


def test_rccl_constants() -> None:
    """Verify RCCL constants."""
    assert RCCL_FLOAT32 == 7
    assert RCCL_FLOAT64 == 8
    assert RCCL_INT32 == 2
    assert RCCL_INT64 == 3
    assert RCCL_SUM == 0
    assert RCCL_PROD == 1
    assert RCCL_MAX == 2
    assert RCCL_MIN == 3


def test_rccl_driver_init_empty_path() -> None:
    """Verify RCCLDriver handles empty lib_path."""
    with patch("ctypes.CDLL", side_effect=OSError("Not found")):
        driver = RCCLDriver(lib_path="")
        assert not driver.is_available()
        assert driver.rccl_lib is None


def test_rccl_driver_init_success() -> None:
    """Verify RCCLDriver initializes successfully when CDLL succeeds."""
    mock_lib = MagicMock()
    with patch("ctypes.CDLL", return_value=mock_lib):
        driver = RCCLDriver(lib_path="/path/to/librccl.so")
        assert driver.is_available()
        assert driver.rccl_lib is mock_lib


def test_rccl_driver_unavailable_branches() -> None:
    """Verify all collectives raise BackendNotSupportedError when driver is unavailable."""
    driver = RCCLDriver()
    driver.rccl_lib = None
    with pytest.raises(BackendNotSupportedError, match="RCCL library is not available"):
        driver.all_reduce(1, 2, 10)
    with pytest.raises(BackendNotSupportedError, match="RCCL library is not available"):
        driver.all_gather(1, 2, 10)
    with pytest.raises(BackendNotSupportedError, match="RCCL library is not available"):
        driver.broadcast(1, 2, 10)
    with pytest.raises(BackendNotSupportedError, match="RCCL library is not available"):
        driver.reduce_scatter(1, 2, 10)
    with pytest.raises(BackendNotSupportedError, match="RCCL library is not available"):
        driver.all_to_all(1, 2, 10)
    with pytest.raises(BackendNotSupportedError, match="RCCL library is not available"):
        driver.get_unique_id()
    with pytest.raises(BackendNotSupportedError, match="RCCL library is not available"):
        driver.init_rank(2, b"x" * 128, 0)
    with pytest.raises(BackendNotSupportedError, match="RCCL library is not available"):
        driver.comm_destroy(123)


def test_rccl_driver_all_reduce() -> None:
    """Verify all_reduce success and error paths."""
    mock_lib = MagicMock()
    mock_lib.ncclAllReduce.return_value = 0

    with patch("ctypes.CDLL", return_value=mock_lib):
        driver = RCCLDriver()
        res1 = driver.all_reduce(100, 200, 4, datatype=RCCL_FLOAT32, op=RCCL_SUM)
        assert res1 == 0

        mock_lib.ncclAllReduce.side_effect = RuntimeError("Failure")
        assert driver.all_reduce(100, 200, 4) == -1


def test_rccl_driver_all_gather() -> None:
    """Verify all_gather success and error paths."""
    mock_lib = MagicMock()
    mock_lib.ncclAllGather.return_value = 0

    with patch("ctypes.CDLL", return_value=mock_lib):
        driver = RCCLDriver()
        res = driver.all_gather(100, 200, 4, datatype=RCCL_FLOAT32)
        assert res == 0

        mock_lib.ncclAllGather.side_effect = RuntimeError("Failure")
        assert driver.all_gather(100, 200, 4) == -1


def test_rccl_driver_broadcast() -> None:
    """Verify broadcast success and error paths."""
    mock_lib = MagicMock()
    mock_lib.ncclBroadcast.return_value = 0

    with patch("ctypes.CDLL", return_value=mock_lib):
        driver = RCCLDriver()
        res = driver.broadcast(100, 200, 4, root=0)
        assert res == 0

        mock_lib.ncclBroadcast.side_effect = RuntimeError("Failure")
        assert driver.broadcast(100, 200, 4) == -1


def test_rccl_driver_reduce_scatter() -> None:
    """Verify reduce_scatter success and error paths."""
    mock_lib = MagicMock()
    mock_lib.ncclReduceScatter.return_value = 0

    with patch("ctypes.CDLL", return_value=mock_lib):
        driver = RCCLDriver()
        res = driver.reduce_scatter(100, 200, 4)
        assert res == 0

        mock_lib.ncclReduceScatter.side_effect = RuntimeError("Failure")
        assert driver.reduce_scatter(100, 200, 4) == -1


def test_rccl_driver_all_to_all() -> None:
    """Verify all_to_all success and error paths."""
    mock_lib = MagicMock()
    mock_lib.ncclAllToAll.return_value = 0

    with patch("ctypes.CDLL", return_value=mock_lib):
        driver = RCCLDriver()
        res = driver.all_to_all(100, 200, 4)
        assert res == 0

        del mock_lib.ncclAllToAll
        del mock_lib.rcclAllToAll
        assert driver.all_to_all(100, 200, 4) == 0


def test_rccl_driver_init_rank_and_destroy() -> None:
    """Verify init_rank and comm_destroy."""
    mock_lib = MagicMock()
    mock_lib.ncclGetUniqueId.return_value = 0
    mock_lib.ncclCommInitRank.return_value = 0
    mock_lib.ncclCommDestroy.return_value = 0

    with patch("ctypes.CDLL", return_value=mock_lib):
        driver = RCCLDriver()
        uid = driver.get_unique_id()
        assert isinstance(uid, bytes)

        comm = driver.init_rank(2, uid, 0)
        assert comm >= 0

        assert driver.comm_destroy(comm) == 0


def test_rccl_driver_missing_functions() -> None:
    """Verify handling when collective functions are missing from RCCL shared library."""
    mock_lib = MagicMock(spec=[])

    with patch("ctypes.CDLL", return_value=mock_lib):
        driver = RCCLDriver()
        assert driver.all_reduce(100, 200, 4) == -1
        assert driver.all_gather(100, 200, 4) == -1
        assert driver.broadcast(100, 200, 4) == -1
        assert driver.reduce_scatter(100, 200, 4) == -1
        assert driver.comm_destroy(123) == -1

        with pytest.raises(RuntimeError, match="ncclGetUniqueId not found"):
            driver.get_unique_id()

        with pytest.raises(RuntimeError, match="ncclCommInitRank not found"):
            driver.init_rank(2, b"x" * 128, 0)


def test_rccl_driver_rccl_prefix_and_explicit_comm_stream() -> None:
    """Verify fallback to rccl-prefixed symbols and explicit communicator/stream arguments."""
    rccl_lib = MagicMock()
    rccl_lib.rcclAllReduce.return_value = 0
    rccl_lib.rcclAllGather.return_value = 0
    rccl_lib.rcclBroadcast.return_value = 0
    rccl_lib.rcclReduceScatter.return_value = 0
    rccl_lib.rcclAllToAll.return_value = 0
    rccl_lib.rcclGetUniqueId.return_value = 0
    rccl_lib.rcclCommInitRank.return_value = 0
    rccl_lib.rcclCommDestroy.return_value = 0
    del rccl_lib.ncclAllReduce
    del rccl_lib.ncclAllGather
    del rccl_lib.ncclBroadcast
    del rccl_lib.ncclReduceScatter
    del rccl_lib.ncclAllToAll
    del rccl_lib.ncclGetUniqueId
    del rccl_lib.ncclCommInitRank
    del rccl_lib.ncclCommDestroy

    with patch("ctypes.CDLL", return_value=rccl_lib):
        driver = RCCLDriver()
        assert driver.all_reduce(100, 200, 4, datatype=RCCL_FLOAT64, op=RCCL_MAX, comm=10, stream=20) == 0
        assert driver.all_gather(100, 200, 4, datatype=RCCL_INT32, comm=10, stream=20) == 0
        assert driver.broadcast(100, 200, 4, root=1, comm=10, stream=20) == 0
        assert driver.reduce_scatter(100, 200, 4, datatype=RCCL_INT64, op=RCCL_MIN, comm=10, stream=20) == 0
        assert driver.all_to_all(100, 200, 4, comm=10, stream=20) == 0
        uid = driver.get_unique_id()
        assert driver.init_rank(2, uid, 0) == 0
        assert driver.comm_destroy(123) == 0


def test_rccl_driver_exceptions_and_status_errors() -> None:
    """Verify exception and non-zero status code handling across driver methods."""
    err_lib = MagicMock()
    err_lib.ncclAllToAll.side_effect = RuntimeError("AllToAll error")
    err_lib.ncclCommDestroy.side_effect = RuntimeError("Destroy error")
    err_lib.ncclGetUniqueId.return_value = 3
    err_lib.ncclCommInitRank.return_value = 4

    with patch("ctypes.CDLL", return_value=err_lib):
        driver = RCCLDriver()
        assert driver.all_to_all(100, 200, 4) == -1
        assert driver.comm_destroy(123) == -1

        with pytest.raises(RuntimeError, match="rcclGetUniqueId failed with status 3"):
            driver.get_unique_id()

        with pytest.raises(RuntimeError, match="rcclCommInitRank failed with status 4"):
            driver.init_rank(2, b"x" * 128, 0)
