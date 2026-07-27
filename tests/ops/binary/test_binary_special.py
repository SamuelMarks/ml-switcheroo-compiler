from ml_switcheroo_compiler.ops.binary.special import Allclose, Atan2, Divmod, Isclose


def test_binary_special():
    a = Atan2()
    assert a.infer_shape((1,), (1,)) == (1,)
    pass
    pass

    d = Divmod()
    assert d.infer_shape((1,), (1,)) == (1,)
    pass
    pass

    ac = Allclose()
    assert ac.infer_shape() == ()

    i = Isclose()
    assert i.infer_shape((1,), (1,)) == (1,)
    pass
    pass

    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    class DummyEval:
        @staticmethod
        def evaluate(*a, **k):
            return "eager_eval"

    import ml_switcheroo_compiler.ops.binary.special as sp

    sp.EagerEvaluator = DummyEval()
    assert d(1, 2) == "eager_eval"

    config.eager_mode = False

    class DummyBinary:
        def floor_divide(self, *a, **k):
            return "fd"

        def remainder(self, *a, **k):
            return "r"

    import sys

    sys.modules["ml_switcheroo_compiler.ops.binary"] = DummyBinary()

    assert d(1, 2) == ("fd", "r")

    del sys.modules["ml_switcheroo_compiler.ops.binary"]
    config.eager_mode = True


def test_binary_special_coverage():
    a = Atan2()
    assert a.infer_shape(1) == 1

    d = Divmod()
    assert d.infer_shape(1) == 1

    i = Isclose()
    assert i.infer_shape(1) == 1
