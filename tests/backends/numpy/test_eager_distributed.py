from unittest.mock import MagicMock, patch

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext, _np_all_gather, _np_all_reduce, _np_all_to_all, _np_axis_index, _np_broadcast, _np_reduce, _np_reduce_scatter, _np_shard_tensor, set_np_distributed_context


def test_np_axis_index():
    assert _np_axis_index(np) == np.array(0)


def test_tcp_context_init():
    with patch("os.path.exists", return_value=False):
        TCPDistributedContext(world_size=2, rank=0)
    ctx = TCPDistributedContext(world_size=2, rank=0)
    assert ctx.world_size == 2
    assert ctx.rank == 0


def test_tcp_context_initialize():
    ctx = TCPDistributedContext(world_size=1)
    ctx.initialize()  # should return early

    with patch("threading.Thread") as mock_thread, patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Listener") as mock_listener, patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Client") as mock_client:
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        # 1. Test ring
        ctx2 = TCPDistributedContext(world_size=2, rank=0, topology="ring")
        ctx2.initialize()
        # run target of thread
        target = mock_thread.call_args[1].get("target")
        if target:
            target()
        mock_listener.assert_called()
        mock_thread_instance.start.assert_called()
        mock_thread_instance.join.assert_called()

        # 2. Test tree (coverage for topology == "tree")
        ctx3 = TCPDistributedContext(world_size=2, rank=1, topology="tree")
        ctx3.initialize()

        # 3. Test exception in Client connection (coverage for ConnectionRefusedError)
        mock_client.side_effect = [ConnectionRefusedError, ConnectionRefusedError, MagicMock()]
        with patch("time.sleep"):
            ctx4 = TCPDistributedContext(world_size=2, rank=0)
            ctx4.initialize()


def test_tcp_context_all_reduce_ring():
    ctx = TCPDistributedContext(world_size=1)
    assert ctx.all_reduce_ring("tensor", backend_module=np) == "tensor"

    ctx = TCPDistributedContext(world_size=2, rank=0)
    ctx.send_conns = [MagicMock()]
    ctx.recv_conns = [MagicMock()]
    ctx.recv_conns[0].recv.return_value = np.array([2])

    tensor = np.array([1, 1])
    # sum
    res = ctx.all_reduce_ring(tensor, op_type="sum", backend_module=np)
    assert np.array_equal(res, np.array([2, 3]))

    # prod
    ctx.recv_conns[0].recv.return_value = np.array([2])
    res = ctx.all_reduce_ring(tensor, op_type="prod", backend_module=np)

    # max
    res = ctx.all_reduce_ring(tensor, op_type="max", backend_module=np)

    # min
    res = ctx.all_reduce_ring(tensor, op_type="min", backend_module=np)


def test_tcp_context_all_gather_tensors():
    ctx = TCPDistributedContext(world_size=1)
    assert ctx.all_gather_tensors("tensor") == ["tensor"]

    ctx = TCPDistributedContext(world_size=2, rank=0)
    ctx.send_conns = [MagicMock()]
    ctx.recv_conns = [MagicMock()]
    ctx.recv_conns[0].recv.return_value = "recv"

    res = ctx.all_gather_tensors("tensor")
    assert res == ["tensor", "recv"]


def test_tcp_context_shutdown():
    ctx = TCPDistributedContext(world_size=2, rank=0)
    ctx.recv_conns = [MagicMock()]
    ctx.send_conns = [MagicMock()]
    ctx.listener = MagicMock()

    ctx.shutdown()
    ctx.recv_conns[0].close.assert_called()
    ctx.send_conns[0].close.assert_called()
    ctx.listener.close.assert_called()


def test_set_np_distributed_context():
    with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.TCPDistributedContext") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        set_np_distributed_context(world_size=2, rank=1)
        mock_instance.initialize.assert_called()


def test_np_all_reduce():
    set_np_distributed_context(1, 0)
    tensor = np.array([1])
    assert np.array_equal(_np_all_reduce(np, tensor), tensor)

    with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed._tcp_dist_ctx") as ctx:
        ctx.world_size = 2
        ctx.all_gather_tensors.return_value = [tensor, tensor]
        ctx.all_reduce_ring.return_value = "reduced"
        assert _np_all_reduce(np, tensor) == "reduced"


def test_np_all_gather():
    set_np_distributed_context(1, 0)
    tensor = np.array([1])
    assert np.array_equal(_np_all_gather(np, tensor), np.array([[1]]))

    with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed._tcp_dist_ctx") as ctx:
        ctx.world_size = 2
        ctx.all_gather_tensors.return_value = [np.array([1]), np.array([2])]
        res = _np_all_gather(np, tensor, axis=0)
        assert np.array_equal(res, np.array([1, 2]))


def test_np_broadcast():
    set_np_distributed_context(1, 0)
    tensor = np.array([1])
    assert np.array_equal(_np_broadcast(np, tensor), tensor)

    with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed._tcp_dist_ctx") as ctx:
        ctx.world_size = 2
        ctx.all_gather_tensors.return_value = ["a", "b"]
        assert _np_broadcast(np, tensor, root_rank=1) == "b"


def test_np_reduce_scatter():
    set_np_distributed_context(1, 0)
    tensor = np.array([1])
    assert np.array_equal(_np_reduce_scatter(np, tensor), tensor)

    with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed._tcp_dist_ctx") as ctx:
        ctx.world_size = 2
        ctx.rank = 0

        t1 = np.array([1, 2])
        t2 = np.array([3, 4])
        ctx.all_gather_tensors.return_value = [t1, t2]

        # sum
        res = _np_reduce_scatter(np, tensor, op_type="sum")
        assert np.array_equal(res, np.array([4]))

        # prod
        res = _np_reduce_scatter(np, tensor, op_type="prod")

        # max
        res = _np_reduce_scatter(np, tensor, op_type="max")

        # min
        res = _np_reduce_scatter(np, tensor, op_type="min")

        # default
        res = _np_reduce_scatter(np, tensor, op_type="other")


def test_np_reduce():
    set_np_distributed_context(1, 0)
    tensor = np.array([1])
    assert np.array_equal(_np_reduce(np, tensor), tensor)

    with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed._tcp_dist_ctx") as ctx:
        ctx.world_size = 2
        ctx.rank = 0

        t1 = np.array([1, 2])
        t2 = np.array([3, 4])
        ctx.all_gather_tensors.return_value = [t1, t2]

        # sum
        res = _np_reduce(np, tensor, root_rank=0, op_type="sum")
        assert np.array_equal(res, np.array([4, 6]))

        # not root rank
        ctx.rank = 1
        res = _np_reduce(np, tensor, root_rank=0, op_type="sum")
        assert res is None
        ctx.rank = 0

        # prod
        res = _np_reduce(np, tensor, op_type="prod")

        # max
        res = _np_reduce(np, tensor, op_type="max")

        # min
        res = _np_reduce(np, tensor, op_type="min")

        # default
        res = _np_reduce(np, tensor, op_type="other")


def test_np_all_to_all():
    set_np_distributed_context(1, 0)
    tensor = np.array([1])
    assert np.array_equal(_np_all_to_all(np, tensor), tensor)

    with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed._tcp_dist_ctx") as ctx:
        ctx.world_size = 2
        ctx.all_gather_tensors.return_value = ["a", "b"]
        assert _np_all_to_all(np, tensor) == ["a", "b"]


def test_np_shard_tensor():
    assert np.array_equal(_np_shard_tensor(np, [1, 2]), np.array([1, 2]))
