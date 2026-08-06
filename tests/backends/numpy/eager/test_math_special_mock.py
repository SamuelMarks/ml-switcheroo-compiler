import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import _np_rawmatmul


def test_mock(monkeypatch):
    import ml_switcheroo_compiler.ops as ops

    class FakeOp:
        def __init__(self, *args, **kwargs):
            self.hit = True

    monkeypatch.setattr(ops, "RawMatMul", FakeOp, raising=False)

    try:
        res = _np_rawmatmul(np, 1, 2)
        print("RES:", res)
    except Exception as e:
        print("EXCEPTION:", repr(e))
