"""Tests for MLX distributed ops."""

import pytest

try:
    import mlx.core as mx

    has_mlx = True
except ImportError:
    has_mlx = False


@pytest.mark.skipif(not has_mlx, reason="MLX is not installed")
def test_mlx_eager_distributed_ops():
    """Test MLX eager distributed operations."""
    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_all_gather, _mlx_all_reduce, _mlx_all_to_all, _mlx_reduce_scatter

    t = mx.array([1.0, 2.0, 3.0])

    # Should safely fallback or execute
    _mlx_all_reduce(mx, t)
    _mlx_all_gather(mx, t)
    _mlx_all_to_all(mx, t)
    _mlx_reduce_scatter(mx, t)


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
