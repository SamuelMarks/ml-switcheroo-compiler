import importlib


def test_unary_init():
    import ml_switcheroo_compiler.ops.unary as unary

    assert unary.abs is not None


def test_unary_init_keyerror(monkeypatch):
    import sys

    import ml_switcheroo_compiler.ops.registry as registry
    import ml_switcheroo_compiler.ops.unary as unary

    original_get_op = registry.get_op

    def mock_get_op(name):
        if name in ["ModifiedBesselI0", "ModifiedBesselI1", "ModifiedBesselK0", "ModifiedBesselK1"]:
            raise KeyError(name)
        return original_get_op(name)

    actual_base = sys.modules["ml_switcheroo_compiler.ops.base"]
    monkeypatch.setattr(actual_base, "get_op", mock_get_op)

    importlib.reload(unary)

    assert unary.modified_bessel_i0 is None
    assert unary.modified_bessel_i1 is None
    assert unary.modified_bessel_k0 is None
    assert unary.modified_bessel_k1 is None
