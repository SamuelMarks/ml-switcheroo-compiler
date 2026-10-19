"""Tests for test_distributed_ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

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

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def _test_distributed_ops_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_distributed_ops.

    Args:
        _state: Shared test state dictionary.
    """
    import ml_switcheroo_compiler.backends.registry as reg
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    class DummyBackend:
        """Test class DummyBackend."""

        def execute_op(self, __op: EagerValue, *a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function execute_op.

            Args:
                __op: Test parameter `__op`.
                a: Test parameter `a`.
                k: Test parameter `k`.
            """

            class T:
                """Test class T."""

                shape = (1,)
                dtype = "float32"

            return T()

    import ml_switcheroo_compiler.ops.distributed_ops as dist

    orig_dist_backend = cast("EagerTensorProtocol", getattr(dist, "get_active_backend"))
    orig_reg_backend = getattr(reg, "_ACTIVE_BACKEND", None)
    has_reg_backend = hasattr(reg, "_ACTIVE_BACKEND")
    setattr(dist, "get_active_backend", lambda: DummyBackend())
    setattr(reg, "_ACTIVE_BACKEND", DummyBackend())

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data="data", config=TensorConfig((1,), type("D", (), {"value": "float32"})(), "cpu"))

    assert cast("EagerTensorProtocol", getattr(shard_tensor(t, None, None), "shape")) == (1,)
    assert cast("EagerTensorProtocol", getattr(nccl_all_reduce(t), "shape")) == (1,)
    getattr(_state, "update")({"DummyBackend": DummyBackend, "Tensor": Tensor, "TensorConfig": TensorConfig, "config": config})
    getattr(_state, "update")({"dist": dist, "has_reg_backend": has_reg_backend, "orig_dist_backend": orig_dist_backend, "orig_reg_backend": orig_reg_backend})
    getattr(_state, "update")({"reg": reg, "t": t})
    _state.update(locals())


def _test_distributed_ops_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_distributed_ops.

    Args:
        _state: Shared test state dictionary.
    """
    t = _state["t"]
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", hierarchical_copy_all_reduce)(t), "shape")) == (1,)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", broadcast)(t), "shape")) == (1,)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", all_gather)(t), "shape")) == (1,)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", reduce)(t), "shape")) == (1,)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", all_reduce)(t), "shape")) == (1,)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", reduce_scatter)(t), "shape")) == (1,)
    assert cast("EagerTensorProtocol", getattr(all_to_all(t), "shape")) == (1,)
    assert cast("EagerTensorProtocol", getattr(broadcast_arrays(t), "shape")) == (1,)
    assert cast("EagerTensorProtocol", getattr(broadcast_to(t), "shape")) == (1,)
    assert cast("EagerTensorProtocol", getattr(broadcast_to_rank(t), "shape")) == (1,)
    assert cast("EagerTensorProtocol", getattr(broadcasted_iota(t), "shape")) == (1,)
    assert cast("EagerTensorProtocol", getattr(pbroadcast(t), "shape")) == (1,)

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
    _state.update(locals())


def _test_distributed_ops_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_distributed_ops.

    Args:
        _state: Shared test state dictionary.
    """
    config, t = _state["config"], _state["t"]
    assert BroadcastTo().infer_shape() == ()
    assert BroadcastToRank().infer_shape() == ()
    assert BroadcastedIota().infer_shape() == ()
    assert Pbroadcast().infer_shape() == ()
    assert Pmax().infer_shape(t) == (1,)
    assert Pmin().infer_shape(t) == (1,)
    assert Outfeed().infer_shape(t) == ()
    assert Pshuffle().infer_shape(t) == (1,)
    assert Pswapaxes().infer_shape(t, axis1=0, axis2=0) == (1,)
    assert Ppermute().infer_shape(t) == (1,)
    assert PsumScatter().infer_shape(t) == (None,)

    setattr(config, "eager_mode", False)

    from unittest.mock import patch

    getattr(_state, "update")({"patch": patch})
    _state.update(locals())


def _test_distributed_ops_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_distributed_ops.

    Args:
        _state: Shared test state dictionary.
    """
    config, dist, has_reg_backend, orig_dist_backend = _state["config"], _state["dist"], _state["has_reg_backend"], _state["orig_dist_backend"]
    orig_reg_backend, patch, reg, t = _state["orig_reg_backend"], _state["patch"], _state["reg"], _state["t"]
    with patch("ml_switcheroo_compiler.ops.distributed_ops._emit_shape_node", return_value="emitted"):
        assert cast("EagerCallableProtocol", shard_tensor)(t, None, None) == "emitted"
        assert cast("EagerCallableProtocol", nccl_all_reduce)(t) == "emitted"
        assert cast("EagerCallableProtocol", hierarchical_copy_all_reduce)(t) == "emitted"
        assert cast("EagerCallableProtocol", broadcast)(t) == "emitted"
        assert cast("EagerCallableProtocol", all_gather)(t) == "emitted"
        assert cast("EagerCallableProtocol", reduce)(t) == "emitted"
        assert cast("EagerCallableProtocol", all_reduce)(t) == "emitted"
        assert cast("EagerCallableProtocol", reduce_scatter)(t) == "emitted"
        assert all_to_all(t) == "emitted"
        assert broadcast_arrays(t) == "emitted"
        assert broadcast_to(t) == "emitted"
        assert broadcast_to_rank(t) == "emitted"
        assert broadcasted_iota(t) == "emitted"
        assert pbroadcast(t) == "emitted"

    setattr(config, "eager_mode", True)
    setattr(dist, "get_active_backend", orig_dist_backend)
    if has_reg_backend:
        setattr(reg, "_ACTIVE_BACKEND", orig_reg_backend)
    else:
        del reg._ACTIVE_BACKEND


def test_distributed_ops() -> None:
    """Test function test_distributed_ops."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_distributed_ops_part_1(_state)
    _test_distributed_ops_part_2(_state)
    _test_distributed_ops_part_3(_state)
    _test_distributed_ops_part_4(_state)
    assert len(_state) > 0
