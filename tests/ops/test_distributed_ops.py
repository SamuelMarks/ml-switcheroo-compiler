from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.distributed_ops import (
    AllGather,
    AllReduce,
    AllToAll,
    Broadcast,
    BroadcastArrays,
    BroadcastedIota,
    BroadcastTo,
    BroadcastToRank,
    HierarchicalCopyAllReduce,
    NcclAllReduce,
    Outfeed,
    Pbroadcast,
    Pmax,
    Pmin,
    Ppermute,
    Pshuffle,
    PsumScatter,
    Pswapaxes,
    Reduce,
    ReduceScatter,
    ShardTensor,
    all_gather,
    all_reduce,
    all_to_all,
    broadcast,
    broadcast_arrays,
    broadcast_to,
    broadcast_to_rank,
    broadcasted_iota,
    hierarchical_copy_all_reduce,
    nccl_all_reduce,
    pbroadcast,
    recv,
    reduce,
    reduce_scatter,
    send,
    shard_tensor,
)


class MockArray:
    def __init__(self, shape):
        self.shape = tuple(shape)


def test_distributed_ops_infer_shape():
    op1 = ShardTensor()
    assert op1.infer_shape(MockArray((2, 2))) == (2, 2)
    assert op1.infer_shape(None) == ()

    op2 = NcclAllReduce()
    assert op2.infer_shape(MockArray((2, 2))) == (2, 2)
    assert op2.infer_shape(None) == ()

    op3 = HierarchicalCopyAllReduce()
    assert op3.infer_shape(MockArray((2, 2))) == (2, 2)
    assert op3.infer_shape(None) == ()

    op4 = Broadcast()
    assert op4.infer_shape(MockArray((2, 2))) == (2, 2)
    assert op4.infer_shape(None) == ()

    op5 = AllGather()
    assert op5.infer_shape(MockArray((2, 2))) == (2, 2)
    assert op5.infer_shape(None) == ()

    op6 = Reduce()
    assert op6.infer_shape(MockArray((2, 2))) == (2, 2)
    assert op6.infer_shape(None) == ()

    op7 = AllReduce()
    assert op7.infer_shape(MockArray((2, 2))) == (2, 2)
    assert op7.infer_shape(None) == ()

    op8 = ReduceScatter()
    assert op8.infer_shape(MockArray((2, 2))) == (2, 2)
    assert op8.infer_shape(None) == ()

    op9 = AllToAll()
    assert op9.infer_shape(MockArray((2, 2)), MockArray((2, 1))) == (2, 2)
    assert op9.infer_shape() == ()

    op10 = BroadcastArrays()
    assert op10.infer_shape(MockArray((2, 2)), MockArray((2, 1))) == (2, 2)
    assert op10.infer_shape() == ()

    op11 = BroadcastTo()
    assert op11.infer_shape(MockArray((2, 2)), MockArray((2, 1))) == (2, 2)
    assert op11.infer_shape() == ()

    op12 = BroadcastToRank()
    assert op12.infer_shape(MockArray((2, 2)), MockArray((2, 1))) == (2, 2)
    assert op12.infer_shape() == ()

    op13 = BroadcastedIota()
    assert op13.infer_shape(MockArray((2, 2)), MockArray((2, 1))) == (2, 2)
    assert op13.infer_shape() == ()

    op14 = Pbroadcast()
    assert op14.infer_shape(MockArray((2, 2)), MockArray((2, 1))) == (2, 2)
    assert op14.infer_shape() == ()

    op15 = Pmax()
    assert op15.infer_shape(MockArray((2, 2)), MockArray((2, 1))) == (2, 2)
    assert op15.infer_shape() == ()

    op16 = Pmin()
    assert op16.infer_shape(MockArray((2, 2)), MockArray((2, 1))) == (2, 2)
    assert op16.infer_shape() == ()

    op17 = Outfeed()
    assert op17.infer_shape(MockArray((2, 2)), MockArray((2, 1))) == ()
    assert op17.infer_shape() == ()

    op18 = Pshuffle()
    assert op18.infer_shape(MockArray((2, 2)), MockArray((2, 1))) == (2, 2)
    assert op18.infer_shape() == ()

    op19 = Pswapaxes()
    assert op19.infer_shape(MockArray((2, 2)), MockArray((2, 1)), axis=0) == (None, 2)

    op20 = Ppermute()
    assert op20.infer_shape(MockArray((2, 2)), MockArray((2, 1))) == (2, 2)
    assert op20.infer_shape() == ()

    op21 = PsumScatter()
    assert op21.infer_shape(MockArray((2, 2)), MockArray((2, 1))) == (None, 2)


def test_distributed_ops_eager():
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.dtype import DType

    class FakeData:
        shape = (2,)
        id = "fake_data"

    t = Tensor(data=FakeData(), config=TensorConfig((2,), DType.Float32, None))
    config.eager_mode = True

    with patch("ml_switcheroo_compiler.ops.distributed_ops.get_active_backend") as mock_get_backend:
        mock_backend = mock_get_backend.return_value

        # We need mock execute_op to return an array that has a shape
        mock_res = MagicMock()
        mock_res.shape = (2,)
        mock_backend.execute_op.return_value = mock_res

        shard_tensor(t, None, None)
        nccl_all_reduce(t, "sum")
        hierarchical_copy_all_reduce(t, "sum")
        broadcast(t, 0)
        all_gather(t, 0)
        reduce(t, 0, "sum")
        all_reduce(t, "sum")
        reduce_scatter(t, "sum", 0)

        all_to_all(t, t)
        broadcast_arrays(t, t)
        broadcast_to(t, t)
        broadcast_to_rank(t, t)
        broadcasted_iota(t, t)
        pbroadcast(t, t)
        # We don't have pmax, pmin functions in distributed_ops! Wait, let's look at the implementation.
        # Oh, in distributed_ops.py, pmax is missing! Let's check `cat -n src/ml_switcheroo_compiler/ops/distributed_ops.py | grep -E "def pmax"`

    config.eager_mode = False


def test_distributed_ops_tracing():
    from unittest.mock import patch

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.dtype import DType

    class FakeData:
        shape = (2,)
        id = "fake_data"

    t = Tensor(data=FakeData(), config=TensorConfig((2,), DType.Float32, None))
    config.eager_mode = False

    with patch("ml_switcheroo_compiler.ops.distributed_ops._emit_shape_node", return_value="dummy_node") as mock_emit:
        assert shard_tensor(t, None, None) == "dummy_node"
        assert nccl_all_reduce(t, "sum") == "dummy_node"
        assert hierarchical_copy_all_reduce(t, "sum") == "dummy_node"
        assert broadcast(t, 0) == "dummy_node"
        assert all_gather(t, 0) == "dummy_node"
        assert reduce(t, 0, "sum") == "dummy_node"
        assert all_reduce(t, "sum") == "dummy_node"
        assert reduce_scatter(t, "sum", 0) == "dummy_node"

        assert all_to_all(t, t) == "dummy_node"
        assert broadcast_arrays(t, t) == "dummy_node"
        assert broadcast_to(t, t) == "dummy_node"
        assert broadcast_to_rank(t, t) == "dummy_node"
        assert broadcasted_iota(t, t) == "dummy_node"
        assert pbroadcast(t, t) == "dummy_node"

        with patch("ml_switcheroo_compiler.tracing.builder.TracingNodeBuilder.emit_tracing_node", return_value="dummy_send_recv"):
            assert send(t, 1) == "dummy_send_recv"
            assert recv((2,), "float32", 0) == "dummy_send_recv"
