"""Tests for distributed collective operations."""

import pickle
from unittest.mock import MagicMock, patch

import numpy as np

from ml_switcheroo_compiler.distributed.collectives import (
    DistributedBarrier,
    _recv_data,
    _send_data,
    all_gather,
    all_reduce,
    broadcast,
    reduce_scatter,
)


def test_send_recv():
    """Test send and recv logic."""
    data = np.array([1, 2, 3])

    conn = MagicMock()
    _send_data(conn, data)

    payload = pickle.dumps(data)
    expected_data = len(payload).to_bytes(8, "big") + payload
    conn.sendall.assert_called_once_with(expected_data)

    # Test recv
    conn2 = MagicMock()
    # Mock recv to return data_len, then chunk, then empty chunk
    conn2.recv.side_effect = [
        len(payload).to_bytes(8, "big"),
        payload,
        b"",
    ]

    res = _recv_data(conn2)
    np.testing.assert_array_equal(res, data)


def test_all_reduce():
    """Test all_reduce."""
    tensor = np.array([1, 2, 3, 4])

    # World size 1
    res = all_reduce(tensor, None, None, 0, 1)
    np.testing.assert_array_equal(res, tensor)

    # World size 2
    next_conn = MagicMock()
    prev_conn = MagicMock()

    with patch("ml_switcheroo_compiler.distributed.collectives._send_data") as mock_send, patch("ml_switcheroo_compiler.distributed.collectives._recv_data") as mock_recv:
        mock_recv.return_value = np.array([10, 20])  # Dummy received chunk

        res = all_reduce(tensor, next_conn, prev_conn, 0, 2)

        # 1 step scatter-reduce, 1 step all-gather
        assert mock_send.call_count == 2
        assert mock_recv.call_count == 2

        assert len(res) == 4


def test_all_gather():
    """Test all_gather."""
    tensor = np.array([1, 2])

    res = all_gather(tensor, None, None, 0, 1)
    np.testing.assert_array_equal(res, tensor)

    next_conn = MagicMock()
    prev_conn = MagicMock()

    with patch("ml_switcheroo_compiler.distributed.collectives._send_data") as mock_send, patch("ml_switcheroo_compiler.distributed.collectives._recv_data") as mock_recv:
        mock_recv.return_value = np.array([3, 4])

        res = all_gather(tensor, next_conn, prev_conn, 0, 2)

        assert mock_send.call_count == 1
        assert mock_recv.call_count == 1

        assert len(res) == 4


def test_reduce_scatter():
    """Test reduce_scatter."""
    tensor = np.array([1, 2, 3, 4])

    res = reduce_scatter(tensor, None, None, 0, 1)
    np.testing.assert_array_equal(res, tensor)

    next_conn = MagicMock()
    prev_conn = MagicMock()

    with patch("ml_switcheroo_compiler.distributed.collectives._send_data") as mock_send, patch("ml_switcheroo_compiler.distributed.collectives._recv_data") as mock_recv:
        mock_recv.return_value = np.array([10, 20])

        res = reduce_scatter(tensor, next_conn, prev_conn, 0, 2)

        assert mock_send.call_count == 1
        assert mock_recv.call_count == 1

        assert len(res) == 2


def test_broadcast():
    """Test broadcast."""
    tensor = np.array([1, 2])

    conn1 = MagicMock()
    conn2 = MagicMock()

    with patch("ml_switcheroo_compiler.distributed.collectives._send_data") as mock_send, patch("ml_switcheroo_compiler.distributed.collectives._recv_data") as mock_recv:
        # I am root
        res = broadcast(tensor, 0, [conn1, conn2], 0)
        assert mock_send.call_count == 2
        np.testing.assert_array_equal(res, tensor)

        # I am NOT root
        mock_recv.return_value = np.array([3, 4])
        res = broadcast(tensor, 0, [conn1, conn2], 1)
        assert mock_recv.call_count == 1
        np.testing.assert_array_equal(res, np.array([3, 4]))


def test_distributed_barrier():
    """Test DistributedBarrier."""
    barrier = DistributedBarrier(0, 1, [])
    barrier.wait()  # Should return immediately

    conn = MagicMock()
    barrier = DistributedBarrier(0, 2, [conn])

    with patch("ml_switcheroo_compiler.distributed.collectives._send_data") as mock_send, patch("ml_switcheroo_compiler.distributed.collectives._recv_data") as mock_recv:
        barrier.wait(timeout=1.0)
        assert mock_send.call_count == 1
        assert mock_recv.call_count == 1

        # Check timeout was set and unset
        conn.settimeout.assert_any_call(1.0)
        conn.settimeout.assert_any_call(None)


def test_recv_break():
    import pickle

    data = np.array([1, 2, 3])
    payload = pickle.dumps(data)

    conn = MagicMock()
    conn.recv.side_effect = [
        (len(payload) + 10).to_bytes(8, "big"),  # fake longer length
        payload,
        b"",  # triggers break
    ]

    try:
        _recv_data(conn)
    except BaseException:
        pass
