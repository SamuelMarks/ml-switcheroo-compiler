def test_mlx_types():
    import pytest

    try:
        import mlx.core as mx  # noqa: F401
    except ImportError:
        pytest.skip("MLX not available")

    from ml_switcheroo_compiler.backends.mlx.types import array, asarray, item, zeros

    res = zeros(None, (2, 2))
    assert res.shape == (2, 2)

    res = array(None, [1.0, 2.0])
    assert res.shape == (2,)

    res = asarray(None, [1.0, 2.0])
    assert res.shape == (2,)

    res = item(None, array(None, 42.0))
    assert res == 42.0


def test_mlx_types_asarray_fallback(monkeypatch):
    import ml_switcheroo_compiler.backends.mlx.types as types_mod
    from ml_switcheroo_compiler.backends.mlx.types import asarray

    class MockMX:
        def array(self, *args, **kwargs):
            return "array_called"

    monkeypatch.setattr(types_mod, "mx", MockMX())
    res = asarray(None, [1.0])
    assert res == "array_called"


def test_mlx_types_asarray_success(monkeypatch):
    import ml_switcheroo_compiler.backends.mlx.types as types_mod
    from ml_switcheroo_compiler.backends.mlx.types import asarray

    class MockMXWithAsarray:
        def asarray(self, *args, **kwargs):
            return "asarray_called"

    monkeypatch.setattr(types_mod, "mx", MockMXWithAsarray())
    res = asarray(None, [1.0])
    assert res == "asarray_called"
