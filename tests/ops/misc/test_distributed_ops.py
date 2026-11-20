"""Tests for test_distributed_ops."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.core.config import Config
from ml_switcheroo_compiler.core.tensor import Tensor
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
from tests.testing_utils import (
    EagerBoundCallable,
    extract_test_shape,
    require_callable,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class _TestDistributedOpsState(typing.TypedDict, total=False):
    """Typed shared state for `_test_distributed_ops` split helpers."""

    config: Config
    mp: pytest.MonkeyPatch
    patch: EagerBoundCallable
    t: Tensor


def _test_distributed_ops_part_1(_state: _TestDistributedOpsState) -> None:
    """Execute part 1 of test_distributed_ops.

    Args:
        _state (_TestDistributedOpsState): Shared test state dictionary.
    """
    import ml_switcheroo_compiler.backends.registry as reg
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    class DummyBackend:
        """Test class DummyBackend."""

        def execute_op(self, __op: BackendArg, *a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                __op (BackendArg): Test parameter `__op`.
                *a (BackendArg): Test parameter `a`.
                **k (BackendArg): Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """

            class T:
                """Test class T."""

                shape = (1,)
                dtype = "float32"

            return T()

    import ml_switcheroo_compiler.ops.distributed_ops as dist

    mp = _state["mp"]
    mp.setattr(dist, "get_active_backend", lambda: DummyBackend(), raising=False)
    mp.setattr(reg, "_ACTIVE_BACKEND", DummyBackend(), raising=False)

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data="data", config=TensorConfig((1,), type("D", (), {"value": "float32"})(), "cpu"))

    assert extract_test_shape(shard_tensor(t, None, None)) == (1,)
    assert extract_test_shape(nccl_all_reduce(t)) == (1,)
    _state.update({"config": config, "t": t})


def _test_distributed_ops_part_2(_state: _TestDistributedOpsState) -> None:
    """Execute part 2 of test_distributed_ops.

    Args:
        _state (_TestDistributedOpsState): Shared test state dictionary.
    """
    t = _state["t"]
    assert extract_test_shape(hierarchical_copy_all_reduce(t)) == (1,)
    assert extract_test_shape(broadcast(t)) == (1,)
    assert extract_test_shape(all_gather(t)) == (1,)
    assert extract_test_shape(reduce(t)) == (1,)
    assert extract_test_shape(all_reduce(t)) == (1,)
    assert extract_test_shape(reduce_scatter(t)) == (1,)
    assert extract_test_shape(all_to_all(t)) == (1,)
    assert extract_test_shape(broadcast_arrays(t)) == (1,)
    assert extract_test_shape(broadcast_to(t)) == (1,)
    assert extract_test_shape(broadcast_to_rank(t)) == (1,)
    assert extract_test_shape(broadcasted_iota(t)) == (1,)
    assert extract_test_shape(pbroadcast(t)) == (1,)

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
    _state.update({"t": t})


def _test_distributed_ops_part_3(_state: _TestDistributedOpsState) -> None:
    """Execute part 3 of test_distributed_ops.

    Args:
        _state (_TestDistributedOpsState): Shared test state dictionary.
    """
    config, mp, t = _state["config"], _state["mp"], _state["t"]
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

    mp.setattr(config, "eager_mode", False, raising=False)

    from unittest.mock import patch

    _state.update({"patch": require_callable(patch)})
    _state.update({"config": config, "t": t})


def _test_distributed_ops_part_4(_state: _TestDistributedOpsState) -> None:
    """Execute part 4 of test_distributed_ops.

    Args:
        _state (_TestDistributedOpsState): Shared test state dictionary.
    """
    config, mp, patch, t = _state["config"], _state["mp"], _state["patch"], _state["t"]
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

    mp.undo()
    config.eager_mode = True


def test_distributed_ops() -> None:
    """Test function test_distributed_ops."""
    mp = pytest.MonkeyPatch()
    try:
        _state: _TestDistributedOpsState = {"mp": mp}
        _test_distributed_ops_part_1(_state)
        _test_distributed_ops_part_2(_state)
        _test_distributed_ops_part_3(_state)
        _test_distributed_ops_part_4(_state)
        assert len(_state) > 0
    finally:
        mp.undo()
