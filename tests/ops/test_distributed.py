"""Tests for distributed operations."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.distributed import (
    shard_tensor,
    all_reduce,
    all_gather,
    reduce_scatter,
    ShardTensorOp,
    AllReduceOp,
    AllGatherOp,
    ReduceScatterOp,
)
from unittest.mock import MagicMock, patch
from ml_switcheroo_compiler.tracing.tracer import _tracer


def test_distributed_ops_eager():
    """Test distributed ops eager mode."""
    config.eager_mode = True
    _tracer.is_tracing = False  # Reset just in case
    # In eager mode they raise NotImplementedError since there is no backend by default
    # Or actually if backend is numpy, maybe they fail. Let's mock backend.
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        backend_instance = MagicMock()
        mock_backend.return_value = backend_instance
        backend_instance.execute_op.return_value = [1, 2, 3]
        backend_instance.array.return_value = MagicMock(shape=(3,))

        tensor = Tensor(None, (3,), DType.Float32, "cpu")
        res1 = shard_tensor(tensor, "mesh", "layout")
        res2 = all_reduce(tensor)
        res3 = all_gather(tensor)
        res4 = reduce_scatter(tensor)

        assert res1.shape == (3,)
        assert res2.shape == (3,)
        assert res3.shape == (3,)
        assert res4.shape == (3,)


def test_distributed_ops_tracing():
    """Test distributed ops tracing mode."""
    config.eager_mode = False
    _tracer.start_tracing("test_graph")
    try:
        tensor = Tensor(MagicMock(id="t1"), (3,), DType.Float32, "cpu")

        res1 = shard_tensor(tensor, "mesh", "layout")
        res2 = all_reduce(tensor)
        res3 = all_gather(tensor)
        res4 = reduce_scatter(tensor)

        assert res1.shape == ()
        assert res2.shape == ()
        assert res3.shape == ()
        assert res4.shape == ()
    finally:
        _tracer.stop_tracing()


def test_distributed_op_defs():
    """Test OpDef infer_shape."""
    assert ShardTensorOp().infer_shape(None) == ()
    assert AllReduceOp().infer_shape(None) == ()
    assert AllGatherOp().infer_shape(None) == ()
    assert ReduceScatterOp().infer_shape(None) == ()
