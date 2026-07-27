"""Test module."""


class DummyOps:
    def __getattr__(self, name):
        def _dummy(*args, **kwargs):
            return name

        return _dummy


def test_initializers(monkeypatch):
    import ml_switcheroo_compiler.nn.initializers as init

    monkeypatch.setattr(init, "ops", DummyOps())

    assert init.zeros(None, (1,)) == "zeros"
    assert init.ones(None, (1,)) == "ones"
    assert init.constant(42)(None, (1,)) == "full"
    assert init.uniform(1.0)(None, (1,)) == "zeros"
    assert init.normal(1.0, 0.0)(None, (1,)) == "zeros"
    assert init.truncated_normal(1.0, 0.0, -2.0, 2.0)(None, (1,)) == "zeros"
    assert init.orthogonal(1.0)(None, (1,)) == "zeros"
    assert init.variance_scaling(init.InitializerConfig(scale=1.0, mode="fan_in", distribution="truncated_normal"))(None, (1,)) == "zeros"
    assert init.variance_scaling(init.InitializerConfig(scale=1.0, mode="fan_in", distribution="truncated_normal"))(None, (1,), "float32") == "zeros"

    assert init.glorot_uniform()(None, (1,)) == "zeros"
    assert init.glorot_normal()(None, (1,)) == "zeros"
    assert init.he_uniform()(None, (1,)) == "zeros"
    assert init.he_normal()(None, (1,)) == "zeros"
    assert init.lecun_uniform()(None, (1,)) == "zeros"
    assert init.lecun_normal()(None, (1,)) == "zeros"

    assert init.delta_orthogonal()(None, (1,)) == "zeros"
