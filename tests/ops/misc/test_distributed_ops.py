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
    reduce,
    reduce_scatter,
    shard_tensor,
)


def test_distributed_ops():
    import ml_switcheroo_compiler.backends.registry as reg
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    class DummyBackend:
        def execute_op(self, __op, *a, **k):
            class T:
                shape = (1,)
                dtype = "float32"

            return T()

    import ml_switcheroo_compiler.ops.distributed_ops as dist

    dist.get_active_backend = lambda: DummyBackend()
    reg._ACTIVE_BACKEND = DummyBackend()

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data="data", config=TensorConfig((1,), type("D", (), {"value": "float32"})(), "cpu"))

    assert shard_tensor(t, None, None).shape == (1,)
    assert nccl_all_reduce(t).shape == (1,)
    assert hierarchical_copy_all_reduce(t).shape == (1,)
    assert broadcast(t).shape == (1,)
    assert all_gather(t).shape == (1,)
    assert reduce(t).shape == (1,)
    assert all_reduce(t).shape == (1,)
    assert reduce_scatter(t).shape == (1,)
    assert all_to_all(t).shape == (1,)
    assert broadcast_arrays(t).shape == (1,)
    assert broadcast_to(t).shape == (1,)
    assert broadcast_to_rank(t).shape == (1,)
    assert broadcasted_iota(t).shape == (1,)
    assert pbroadcast(t).shape == (1,)

    assert ShardTensor().infer_shape(t) == (1,)
    assert NcclAllReduce().infer_shape(t) == (1,)
    assert HierarchicalCopyAllReduce().infer_shape(t) == (1,)
    assert Broadcast().infer_shape(t) == (1,)
    assert AllGather().infer_shape(t) == (1,)
    assert Reduce().infer_shape(t) == (1,)
    assert AllReduce().infer_shape(t) == (1,)
    assert ReduceScatter().infer_shape(t) == (1,)
    assert AllToAll().infer_shape() == ()
    assert BroadcastArrays().infer_shape() == ()
    assert BroadcastTo().infer_shape() == ()
    assert BroadcastToRank().infer_shape() == ()
    assert BroadcastedIota().infer_shape() == ()
    assert Pbroadcast().infer_shape() == ()
    assert Pmax().infer_shape(t) == (1,)
    assert Pmin().infer_shape(t) == (1,)
    assert Outfeed().infer_shape(t) == ()
    assert Pshuffle().infer_shape(t) == (1,)
    assert Pswapaxes().infer_shape(t) == (1,)
    assert Ppermute().infer_shape(t) == (1,)
    assert PsumScatter().infer_shape(t) == (1,)

    config.eager_mode = False

    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.distributed_ops._emit_shape_node", return_value="emitted"):
        assert shard_tensor(t, None, None) == "emitted"
        assert nccl_all_reduce(t) == "emitted"
        assert hierarchical_copy_all_reduce(t) == "emitted"
        assert broadcast(t) == "emitted"
        assert all_gather(t) == "emitted"
        assert reduce(t) == "emitted"
        assert all_reduce(t) == "emitted"
        assert reduce_scatter(t) == "emitted"
        assert all_to_all(t) == "emitted"
        assert broadcast_arrays(t) == "emitted"
        assert broadcast_to(t) == "emitted"
        assert broadcast_to_rank(t) == "emitted"
        assert broadcasted_iota(t) == "emitted"
        assert pbroadcast(t) == "emitted"

    config.eager_mode = True
