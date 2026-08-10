from unittest.mock import MagicMock

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.numpy.eager.distributed as dmod
from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext, _np_all_gather, _np_all_reduce, _np_all_to_all, _np_reduce, _np_reduce_scatter


def test_tcp_dist_ctx_coverage():
    # Test world_size <= 1 return [tensor]
    ctx = TCPDistributedContext(world_size=1)
    assert ctx.all_gather_tensors("test") == ["test"]

    # Test rank 0 gather
    ctx = TCPDistributedContext(world_size=2, rank=0)
    mock_conn = MagicMock()
    mock_conn.recv.return_value = {"rank": 1, "tensor": "B"}
    ctx.connections = [mock_conn]
    res = ctx.all_gather_tensors("A")
    assert res == ["A", "B"]
    mock_conn.send.assert_called_with(["A", "B"])

    # Test rank 0 shutdown
    ctx.listener = MagicMock()
    ctx.shutdown()
    mock_conn.close.assert_called_once()
    ctx.listener.close.assert_called_once()

    # Test rank 1 missing conn
    ctx = TCPDistributedContext(world_size=2, rank=1)
    with pytest.raises(RuntimeError, match="Not initialized"):
        ctx.all_gather_tensors("A")


def test_np_ops_coverage():
    dmod._tcp_dist_ctx.world_size = 2
    dmod._tcp_dist_ctx.rank = 0
    dmod._tcp_dist_ctx.all_gather_tensors = lambda t: [t, np.array([3.0])]

    t = np.array([2.0])
    # cover sum for all ops
    assert _np_all_reduce(np, t, op_type="sum").item() == 5.0
    assert _np_reduce(np, t, op_type="sum").item() == 5.0

    # cover max for all ops
    assert _np_all_reduce(np, t, op_type="max").item() == 3.0

    # cover all_gather concatenate
    res = _np_all_gather(np, t, axis=0)
    assert np.array_equal(res, [2.0, 3.0])

    # cover reduce_scatter sum
    t_rs = np.array([1.0, 2.0])
    dmod._tcp_dist_ctx.all_gather_tensors = lambda t: [t, np.array([3.0, 4.0])]
    assert _np_reduce_scatter(np, t_rs, op_type="sum").item() == 4.0

    # cover all_to_all
    a2a = _np_all_to_all(np, t_rs)
    assert len(a2a) == 2


from unittest.mock import patch


def test_tcp_dist_ctx_connection_refused():
    ctx = TCPDistributedContext(world_size=2, rank=1)

    with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Client") as mock_client:
        # First call raises ConnectionRefusedError, second returns a mock connection
        mock_client.side_effect = [ConnectionRefusedError(), MagicMock()]
        ctx.initialize()

        assert mock_client.call_count == 2
        assert ctx.conn is not None
