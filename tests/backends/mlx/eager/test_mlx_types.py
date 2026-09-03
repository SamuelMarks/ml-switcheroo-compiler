def test_mlx_asarray_fallback(mocker):
    """Test mlx asarray fallback."""
    import ml_switcheroo_compiler.backends.mlx.types as types_mod
    from ml_switcheroo_compiler.backends.mlx.types import asarray

    class DummyMX:
        def array(self, data):
            return "array_" + str(data)

    mocker.patch.object(types_mod, "mx", DummyMX())

    assert asarray(None, "foo") == "array_foo"
