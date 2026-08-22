"""Tests for MLX distributed ops."""

import sys
from unittest.mock import MagicMock


def test_mlx_eager_distributed_ops():
    """Test MLX eager distributed operations."""
    # Mock mlx.core for testing
    mock_mx = MagicMock()
    mock_array = MagicMock()
    mock_mx.array.return_value = mock_array

    # Setup distributed mocks
    mock_mx.distributed.all_sum.return_value = mock_array
    mock_mx.distributed.all_gather.return_value = mock_array
    mock_mx.distributed.all_to_all.return_value = mock_array
    mock_mx.distributed.recv.return_value = mock_array

    sys.modules["mlx"] = MagicMock()
    sys.modules["mlx.core"] = mock_mx

    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_all_gather, _mlx_all_reduce, _mlx_all_to_all, _mlx_reduce_scatter

    t = mock_mx.array([1.0, 2.0, 3.0])

    # Should safely fallback or execute
    _mlx_all_reduce(mock_mx, t)
    _mlx_all_gather(mock_mx, t)
    _mlx_all_to_all(mock_mx, t)
    _mlx_reduce_scatter(mock_mx, t)

    del sys.modules["mlx.core"]
    del sys.modules["mlx"]


def test_mlx_generator_distributed_ops():
    """Test MLX generator AST generation for distributed ops."""
    from ml_switcheroo_compiler.backends.mlx.mlx_mixins import MLXOpRegistryMixin

    mixin = MLXOpRegistryMixin()

    ar = mixin.visit_AllReduce(None, ["tensor"])
    assert "mx.distributed.all_sum(tensor)" in ar

    ag = mixin.visit_AllGather(None, ["tensor"])
    assert "mx.distributed.all_gather(tensor)" in ag

    a2a = mixin.visit_AllToAll(None, ["tensor"])
    assert "mx.distributed.all_to_all(tensor)" in a2a

    rs = mixin.visit_ReduceScatter(None, ["tensor"])
    assert "mx.distributed.recv(tensor)" in rs
