"""Tests for test_distributed_ops."""

from __future__ import annotations

import typing

from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def _check_distributed_infer_shape(classes: EagerValue, t: EagerValue) -> None:
    """Test infer_shape for distributed op classes.

    Args:
        classes: List of class names.
        t: Sample Tensor instance.
    """
    from ml_switcheroo_compiler.ops.registry import get_op

    for cls_name in as_tensor(classes):
        OpCls = as_callable(get_op)(cls_name)
        inst = OpCls()
        if cls_name == "Recv":
            inst.infer_shape((1,), "float32", 0)
        elif cls_name == "Send":
            inst.infer_shape(t, 0)
        else:
            inst.infer_shape(t)


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
        def execute_op(cls, op: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function execute_op.

            Args:
                op: Test parameter `op`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return DummyData()

    orig_eager = config.eager_mode
    try:
        with patch("ml_switcheroo_compiler.backends.registry.BackendRegistry.get", return_value=DummyBackend):
            config.eager_mode = True
            t = Tensor(DummyData(), TensorConfig((1,), "float32", "cpu"))
            for func_name in funcs:
                if func_name not in ("send", "recv"):
                    func = getattr(dist_ops, func_name)
                    func(t, None, None) if func_name == "shard_tensor" else func(t)

            config.eager_mode = False
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

            _check_distributed_infer_shape(classes, t)
    finally:
        config.eager_mode = orig_eager
    assert config.eager_mode == orig_eager


def test_distributed_ops_infer_shapes_extras() -> None:
    """Test function test_distributed_ops_infer_shapes_extras."""
    from ml_switcheroo_compiler.ops.distributed_ops import AllToAll, BroadcastArrays, BroadcastedIota, BroadcastTo, BroadcastToRank, Pbroadcast

    class DummyShape:
        """Test class DummyShape."""

        def __init__(self, shape: EagerValue) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape

    for cls in (AllToAll, BroadcastArrays, BroadcastTo, BroadcastToRank, BroadcastedIota, Pbroadcast):
        op = cls()
        assert op.infer_shape() == ()
        assert as_callable(op.infer_shape)(1, 2) == ()  # no shape attr

        # one shape
        res = as_callable(op.infer_shape)(as_callable(DummyShape)((2,)))
        assert res == (2,) or (cls == BroadcastToRank and res == (2,))

        if cls != BroadcastToRank:
            # two shapes
            assert as_callable(op.infer_shape)(as_callable(DummyShape)((2, 2)), as_callable(DummyShape)((1, 2))) == (2, 2)

    # BroadcastTo specific
    assert BroadcastTo().infer_shape(shape=(3, 3)) == (3, 3)

    # BroadcastToRank specific
    assert as_callable(BroadcastToRank)().infer_shape(as_callable(DummyShape)((2,)), rank=3) == (1, 1, 2)
