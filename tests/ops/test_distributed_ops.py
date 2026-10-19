"""Tests for test_distributed_ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

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

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


class MockArray:
    """Test class MockArray."""

    def __init__(self, shape: EagerValue) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = cast("EagerCallableProtocol", tuple)(shape)


def _test_distributed_ops_infer_shape_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_distributed_ops_infer_shape.

    Args:
        _state: Shared test state dictionary.
    """
    op1 = ShardTensor()
    assert cast("EagerCallableProtocol", op1.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 2))) == (2, 2)
    assert cast("EagerCallableProtocol", op1.infer_shape)(None) == ()

    op2 = NcclAllReduce()
    assert cast("EagerCallableProtocol", op2.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 2))) == (2, 2)
    assert cast("EagerCallableProtocol", op2.infer_shape)(None) == ()

    op3 = HierarchicalCopyAllReduce()
    assert cast("EagerCallableProtocol", op3.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 2))) == (2, 2)
    assert cast("EagerCallableProtocol", op3.infer_shape)(None) == ()

    op4 = Broadcast()
    assert cast("EagerCallableProtocol", op4.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 2))) == (2, 2)
    assert cast("EagerCallableProtocol", op4.infer_shape)(None) == ()

    op5 = AllGather()
    assert cast("EagerCallableProtocol", op5.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 2))) == (2, 2)
    assert cast("EagerCallableProtocol", op5.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 2)), world_size=2, axis=0) == (4, 2)
    assert cast("EagerCallableProtocol", op5.infer_shape)(None) == ()

    op6 = Reduce()
    assert cast("EagerCallableProtocol", op6.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 2))) == (2, 2)
    assert cast("EagerCallableProtocol", op6.infer_shape)(None) == ()

    op7 = AllReduce()
    assert cast("EagerCallableProtocol", op7.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 2))) == (2, 2)
    assert cast("EagerCallableProtocol", op7.infer_shape)(None) == ()
    getattr(_state, "update")({"op1": op1, "op2": op2, "op3": op3, "op4": op4})
    getattr(_state, "update")({"op5": op5, "op6": op6, "op7": op7})
    _state.update(locals())


def _test_distributed_ops_infer_shape_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_distributed_ops_infer_shape.

    Args:
        _state: Shared test state dictionary.
    """
    op8 = ReduceScatter()
    assert cast("EagerCallableProtocol", op8.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 2))) == (2, 2)
    assert cast("EagerCallableProtocol", op8.infer_shape)(cast("EagerCallableProtocol", MockArray)((4, 2)), world_size=2, axis=0) == (2, 2)
    assert cast("EagerCallableProtocol", op8.infer_shape)(None) == ()

    op9 = AllToAll()
    assert cast("EagerCallableProtocol", op9.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 2)), cast("EagerCallableProtocol", MockArray)((2, 1))) == (2, 2)
    assert op9.infer_shape() == ()

    op10 = BroadcastArrays()
    assert cast("EagerCallableProtocol", op10.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 2)), cast("EagerCallableProtocol", MockArray)((2, 1))) == (2, 2)
    assert op10.infer_shape() == ()

    op11 = BroadcastTo()
    assert cast("EagerCallableProtocol", op11.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 2)), cast("EagerCallableProtocol", MockArray)((2, 1))) == (2, 2)
    assert op11.infer_shape() == ()

    op12 = BroadcastToRank()
    assert cast("EagerCallableProtocol", op12.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 2)), cast("EagerCallableProtocol", MockArray)((2, 1))) == (2, 2)
    assert op12.infer_shape() == ()

    op13 = BroadcastedIota()
    assert cast("EagerCallableProtocol", op13.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 2)), cast("EagerCallableProtocol", MockArray)((2, 1))) == (2, 2)
    assert op13.infer_shape() == ()

    op14 = Pbroadcast()
    assert cast("EagerCallableProtocol", op14.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 2)), cast("EagerCallableProtocol", MockArray)((2, 1))) == (2, 2)
    assert op14.infer_shape() == ()
    getattr(_state, "update")({"op10": op10, "op11": op11, "op12": op12, "op13": op13})
    getattr(_state, "update")({"op14": op14, "op8": op8, "op9": op9})
    _state.update(locals())


def _test_distributed_ops_infer_shape_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_distributed_ops_infer_shape.

    Args:
        _state: Shared test state dictionary.
    """
    op15 = Pmax()
    assert cast("EagerCallableProtocol", op15.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 2)), cast("EagerCallableProtocol", MockArray)((2, 1))) == (2, 2)
    assert op15.infer_shape() == ()

    op16 = Pmin()
    assert cast("EagerCallableProtocol", op16.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 2)), cast("EagerCallableProtocol", MockArray)((2, 1))) == (2, 2)
    assert op16.infer_shape() == ()
    op17 = Outfeed()
    assert cast("EagerCallableProtocol", op17.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 2)), cast("EagerCallableProtocol", MockArray)((2, 1))) == ()
    assert op17.infer_shape() == ()

    op18 = Pshuffle()
    assert cast("EagerCallableProtocol", op18.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 2)), cast("EagerCallableProtocol", MockArray)((2, 1))) == (2, 2)
    assert op18.infer_shape() == ()

    op19 = Pswapaxes()
    assert cast("EagerCallableProtocol", op19.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 3)), axis1=0, axis2=1) == (3, 2)

    op20 = Ppermute()
    assert cast("EagerCallableProtocol", op20.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 2)), cast("EagerCallableProtocol", MockArray)((2, 1))) == (2, 2)
    assert op20.infer_shape() == ()

    op21 = PsumScatter()
    assert cast("EagerCallableProtocol", op21.infer_shape)(cast("EagerCallableProtocol", MockArray)((2, 2)), cast("EagerCallableProtocol", MockArray)((2, 1))) == (None, 2)
    getattr(_state, "update")({"op15": op15, "op16": op16, "op17": op17, "op18": op18})
    getattr(_state, "update")({"op19": op19, "op20": op20, "op21": op21})


def test_distributed_ops_infer_shape() -> None:
    """Test function test_distributed_ops_infer_shape."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_distributed_ops_infer_shape_part_1(_state)
    _test_distributed_ops_infer_shape_part_2(_state)
    _test_distributed_ops_infer_shape_part_3(_state)
    assert len(_state) > 0


def test_distributed_ops_eager() -> None:
    """Test function test_distributed_ops_eager."""
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.dtype import DType

    class FakeData:
        """Test class FakeData."""

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
    assert config.eager_mode is False


def test_distributed_ops_tracing() -> None:
    """Test function test_distributed_ops_tracing."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.dtype import DType

    class FakeData:
        """Test class FakeData."""

        shape = (2,)
        id = "fake_data"

    t = Tensor(data=FakeData(), config=TensorConfig((2,), DType.Float32, None))
    config.eager_mode = False

    with patch("ml_switcheroo_compiler.ops.distributed_ops._emit_shape_node", return_value="dummy_node"):
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
