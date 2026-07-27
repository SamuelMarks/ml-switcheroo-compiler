"""Test mlx coverage."""

from ml_switcheroo_compiler.backends.mlx.types import array, asarray, item, zeros


def test_mlx_types() -> None:
    """Test mlx types."""
    assert array is not None
    assert asarray is not None
    assert item is not None
    assert zeros is not None
    import mlx.core as mx

    array(None, [1, 2])
    asarray(None, [3, 4])
    zeros(None, (2,))
    item(None, mx.array([5]))


def test_mlx_types_asarray_mock() -> None:
    """Test mlx asarray mock."""
    import mlx.core as mx

    from ml_switcheroo_compiler.backends.mlx.types import asarray

    mx.asarray = lambda x: mx.array(x)
    asarray(None, [1, 2])
    del mx.asarray
