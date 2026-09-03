from unittest.mock import MagicMock, patch

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext, _np_all_gather, _np_all_reduce, _np_all_to_all, _np_axis_index, _np_broadcast, _np_reduce, _np_reduce_scatter, _np_shard_tensor, set_np_distributed_context


def test_axis_index():
    assert _np_axis_index(np) == 0


def test_tcp_dist_ctx_init():
    with patch("os.path.exists", return_value=False):
        ctx = TCPDistributedContext(topology="tree")
        assert ctx.config == {}


def test_tcp_dist_ctx_initialize_tree():
    with patch("os.path.exists", return_value=False):
        ctx = TCPDistributedContext(world_size=2, rank=1, topology="tree", port=45000)
        # We just need coverage, so we don't need real sockets if we mock
        with patch("threading.Thread") as mock_thread:
            with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Listener") as mock_listener:
                mock_listener.return_value.accept.return_value = "conn"
                with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Client", side_effect=ConnectionRefusedError):
                    with patch("time.sleep"):
                        ctx.initialize()

        ctx = TCPDistributedContext(world_size=2, rank=0, topology="unknown", port=45000)
        with patch("threading.Thread") as mock_thread:
            with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Listener") as mock_listener:
                mock_listener.return_value.accept.return_value = "conn"
                with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Client"):
                    ctx.initialize()


def test_tcp_dist_ctx_all_reduce_ops():
    ctx = TCPDistributedContext(world_size=2, rank=0)

    # Mocking connections
    mock_send = MagicMock()
    mock_recv = MagicMock()

    ctx.send_conns = [mock_send]
    ctx.recv_conns = [mock_recv]

    arr = np.array([1.0, 2.0])

    for op in ["sum", "prod", "max", "min"]:
        mock_recv.recv.return_value = np.array([2.0])
        res = ctx.all_reduce_ring(arr, op_type=op, backend_module=np)
        assert res is not None

    ctx1 = TCPDistributedContext(world_size=1)
    assert ctx1.all_reduce_ring(arr) is arr

    ctx2 = TCPDistributedContext(world_size=2, rank=0)
    ctx2.send_conns = [mock_send]
    ctx2.recv_conns = [mock_recv]
    mock_recv.recv.return_value = np.array([1.0, 2.0])
    res_gather = ctx2.all_gather_tensors(arr)
    assert len(res_gather) == 2
    assert ctx1.all_gather_tensors(arr) == [arr]


def test_tcp_dist_ctx_shutdown():
    ctx = TCPDistributedContext(world_size=2)
    mock_send = MagicMock()
    mock_recv = MagicMock()
    mock_listener = MagicMock()

    ctx.send_conns = [mock_send]
    ctx.recv_conns = [mock_recv]
    ctx.listener = mock_listener

    ctx.shutdown()
    assert mock_send.close.called
    assert mock_recv.close.called
    assert mock_listener.close.called


def test_np_ops_coverage():
    with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.TCPDistributedContext.initialize"):
        set_np_distributed_context(world_size=2, rank=0)

    with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.TCPDistributedContext.all_gather_tensors", return_value=[np.array([1.0]), np.array([2.0])]):
        with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.TCPDistributedContext.all_reduce_ring", return_value=np.array([3.0])):
            _np_all_reduce(np, np.array([1.0]), op_type="sum")

        _np_all_gather(np, np.array([1.0]), axis=0)
        _np_broadcast(np, np.array([1.0]), root_rank=1)

        for op in ["sum", "prod", "max", "min", "unknown"]:
            _np_reduce_scatter(np, np.array([1.0, 2.0]), op_type=op, axis=0)
            _np_reduce(np, np.array([1.0]), root_rank=0, op_type=op)

        _np_all_to_all(np, np.array([1.0, 2.0]))
        _np_shard_tensor(np, np.array([1.0]))
