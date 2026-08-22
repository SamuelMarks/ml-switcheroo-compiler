from unittest.mock import MagicMock, patch


def test_mlx_eager_distributed_all_reduce_fallback():
    import mlx.core as mx

    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_all_reduce

    with patch("builtins.hasattr", return_value=False):
        res = _mlx_all_reduce(mx, mx.array([1.0]))
        assert res.tolist() == [1.0]


def test_mlx_eager_distributed_all_gather_fallback():
    import mlx.core as mx

    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_all_gather

    with patch("builtins.hasattr", return_value=False):
        res = _mlx_all_gather(mx, mx.array([1.0]))
        assert res.shape == (1, 1)


def test_mlx_eager_distributed_all_to_all():
    import mlx.core as mx

    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_all_to_all

    with patch("builtins.hasattr", return_value=True):
        mock_dist = MagicMock()
        mock_dist.all_to_all.return_value = "mocked_all_to_all"

        with patch("mlx.core.distributed", mock_dist, create=True):
            res = _mlx_all_to_all(mx, mx.array([1.0]))
            assert res == "mocked_all_to_all"
