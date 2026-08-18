def test_all_distributed_ops():
    from unittest.mock import patch

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    funcs = [
        "shard_tensor",
        "nccl_all_reduce",
        "hierarchical_copy_all_reduce",
        "broadcast",
        "all_gather",
        "reduce",
        "all_reduce",
        "reduce_scatter",
        "all_to_all",
        "broadcast_arrays",
        "broadcast_to",
        "broadcast_to_rank",
        "broadcasted_iota",
        "pbroadcast",
        "outfeed",
        "pmax",
        "pmin",
        "ppermute",
        "pshuffle",
        "psum_scatter",
        "pswapaxes",
        "send",
        "recv",
    ]
    classes = [
        "ShardTensor",
        "NcclAllReduce",
        "HierarchicalCopyAllReduce",
        "Broadcast",
        "AllGather",
        "Reduce",
        "AllReduce",
        "ReduceScatter",
        "AllToAll",
        "BroadcastArrays",
        "BroadcastTo",
        "BroadcastToRank",
        "BroadcastedIota",
        "Pbroadcast",
        "Pmax",
        "Pmin",
        "Outfeed",
        "Pshuffle",
        "Pswapaxes",
        "Ppermute",
        "PsumScatter",
        "Send",
        "Recv",
    ]

    import ml_switcheroo_compiler.ops.distributed_ops as dist_ops

    class DummyData:
        shape = (1,)
        dtype = "float32"
        id = "dummy"

    class DummyBackend:
        @classmethod
        def execute_op(cls, op, *args, **kwargs):
            return DummyData()

    orig_eager = config.eager_mode
    try:
        with patch("ml_switcheroo_compiler.backends.registry.BackendRegistry.get", return_value=DummyBackend):
            config.eager_mode = True

            t = Tensor(DummyData(), TensorConfig((1,), "float32", "cpu"))

            for func_name in funcs:
                if func_name in ("send", "recv"):
                    continue
                func = getattr(dist_ops, func_name)
                if func_name == "shard_tensor":
                    func(t, None, None)
                else:
                    func(t)

            # Test lazy
            config.eager_mode = False
            from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, global_tracing_state

            global_tracing_state.start_tracing()
            try:
                pt = ProxyTensor(id="pt", shape=(1,), dtype="float32")
                for func_name in funcs:
                    func = getattr(dist_ops, func_name)
                    if func_name == "shard_tensor":
                        func(pt, None, None)
                    elif func_name == "recv":
                        func((1,), "float32", 0)
                    elif func_name == "send":
                        func(pt, 0)
                    else:
                        func(pt)
            finally:
                global_tracing_state.stop_tracing()

            # Test infer shape
            from ml_switcheroo_compiler.ops.registry import get_op

            for cls_name in classes:
                OpCls = get_op(cls_name)
                inst = OpCls()
                if cls_name == "Recv":
                    inst.infer_shape((1,), "float32", 0)
                elif cls_name == "Send":
                    inst.infer_shape(t, 0)
                elif cls_name == "ShardTensor":
                    inst.infer_shape(t)
                else:
                    inst.infer_shape(t)

    finally:
        config.eager_mode = orig_eager


def test_distributed_ops_infer_shapes_extras():
    from ml_switcheroo_compiler.ops.distributed_ops import AllToAll, BroadcastArrays, BroadcastedIota, BroadcastTo, BroadcastToRank, Pbroadcast

    class DummyShape:
        def __init__(self, shape):
            self.shape = shape

    for cls in (AllToAll, BroadcastArrays, BroadcastTo, BroadcastToRank, BroadcastedIota, Pbroadcast):
        op = cls()
        assert op.infer_shape() == ()
        assert op.infer_shape(1, 2) == ()  # no shape attr

        # one shape
        res = op.infer_shape(DummyShape((2,)))
        assert res == (2,) or (cls == BroadcastToRank and res == (2,))

        if cls != BroadcastToRank:
            # two shapes
            assert op.infer_shape(DummyShape((2, 2)), DummyShape((1, 2))) == (2, 2)

    # BroadcastTo specific
    assert BroadcastTo().infer_shape(shape=(3, 3)) == (3, 3)

    # BroadcastToRank specific
    assert BroadcastToRank().infer_shape(DummyShape((2,)), rank=3) == (1, 1, 2)
