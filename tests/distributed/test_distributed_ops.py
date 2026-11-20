"""Tests for test_distributed_ops."""

from __future__ import annotations

import typing

from tests.testing_utils import MockTestStub, require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def _check_distributed_infer_shape(classes: list[str], t: BackendArg) -> None:
    """Test infer_shape for distributed op classes.

    Args:
        classes: List of class names.
        t: Sample Tensor instance.
    """
    from ml_switcheroo_compiler.ops.registry import get_op

    for cls_name in classes:
        OpCls = get_op(cls_name)
        inst = OpCls()
        infer_fn = require_callable(inst.infer_shape)
        if cls_name == "Recv":
            infer_fn((1,), "float32", 0)
        elif cls_name == "Send":
            infer_fn(t, 0)
        else:
            infer_fn(t)


def test_all_distributed_ops() -> None:
    """Test function test_all_distributed_ops."""
    from unittest.mock import patch

    import ml_switcheroo_compiler.ops.distributed_ops as dist_ops
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, global_tracing_state

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

    class DummyData:
        """Test class DummyData."""

        shape = (1,)
        dtype = "float32"
        id = "dummy"

    class DummyBackend:
        """Test class DummyBackend."""

        @classmethod
        def execute_op(cls, op: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                op: Test parameter `op`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """
            return DummyData()

    dist_members = {
        "shard_tensor": dist_ops.shard_tensor,
        "nccl_all_reduce": dist_ops.nccl_all_reduce,
        "hierarchical_copy_all_reduce": dist_ops.hierarchical_copy_all_reduce,
        "broadcast": dist_ops.broadcast,
        "all_gather": dist_ops.all_gather,
        "reduce": dist_ops.reduce,
        "all_reduce": dist_ops.all_reduce,
        "reduce_scatter": dist_ops.reduce_scatter,
        "all_to_all": dist_ops.all_to_all,
        "broadcast_arrays": dist_ops.broadcast_arrays,
        "broadcast_to": dist_ops.broadcast_to,
        "broadcast_to_rank": dist_ops.broadcast_to_rank,
        "broadcasted_iota": dist_ops.broadcasted_iota,
        "pbroadcast": dist_ops.pbroadcast,
        "outfeed": dist_ops.outfeed,
        "pmax": dist_ops.pmax,
        "pmin": dist_ops.pmin,
        "ppermute": dist_ops.ppermute,
        "pshuffle": dist_ops.pshuffle,
        "psum_scatter": dist_ops.psum_scatter,
        "pswapaxes": dist_ops.pswapaxes,
        "send": dist_ops.send,
        "recv": dist_ops.recv,
    }
    orig_eager = config.eager_mode
    try:
        with patch("ml_switcheroo_compiler.backends.registry.BackendRegistry.get", return_value=DummyBackend):
            config.eager_mode = True
            t = Tensor(DummyData(), TensorConfig((1,), "float32", "cpu"))
            for func_name in funcs:
                if func_name not in ("send", "recv"):
                    func = dist_members[func_name]
                    require_callable(func)(t, None, None) if func_name == "shard_tensor" else require_callable(func)(t)

            config.eager_mode = False
            global_tracing_state.start_tracing()
            try:
                pt = ProxyTensor(id="pt", shape=(1,), dtype="float32")
                for func_name in funcs:
                    func = dist_members[func_name]
                    if func_name == "shard_tensor":
                        require_callable(func)(pt, None, None)
                    elif func_name == "recv":
                        require_callable(func)((1,), "float32", 0)
                    elif func_name == "send":
                        require_callable(func)(pt, 0)
                    else:
                        require_callable(func)(pt)
            finally:
                global_tracing_state.stop_tracing()

            _check_distributed_infer_shape(classes, t)
    finally:
        config.eager_mode = orig_eager
    assert config.eager_mode == orig_eager


def test_distributed_ops_infer_shapes_extras() -> None:
    """Test function test_distributed_ops_infer_shapes_extras."""
    from ml_switcheroo_compiler.ops.distributed_ops import (
        AllToAll,
        BroadcastArrays,
        BroadcastedIota,
        BroadcastTo,
        BroadcastToRank,
        Pbroadcast,
    )

    class DummyShape(MockTestStub):
        """Test class DummyShape."""

        def __init__(self, shape: BackendArg) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
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
