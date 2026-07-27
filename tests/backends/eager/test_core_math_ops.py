import numpy as np

import ml_switcheroo_compiler.backends.eager.core_math_ops as core_math_ops


def test_core_math_ops_coverage():
    # TrueDivide
    res = core_math_ops._true_divide(np, 10.0, 2.0)
    assert res == 5.0
    res = core_math_ops._true_divide(object, 10.0, 2.0)
    assert res is None

    # Fft, Rfft, Fftn
    res = core_math_ops._fft(np, np.array([1.0, 2.0]))
    assert res is not None
    res = core_math_ops._rfft(np, np.array([1.0, 2.0]))
    assert res is not None
    res = core_math_ops._fftn(np, np.array([1.0, 2.0]))
    assert res is not None

    res = core_math_ops._fft(object, np.array([1.0, 2.0]))
    assert res is None
    res = core_math_ops._rfft(object, np.array([1.0, 2.0]))
    assert res is None
    res = core_math_ops._fftn(object, np.array([1.0, 2.0]))
    assert res is None

    # Erf, Erfc, Expm1, Erfinv
    res = core_math_ops._erf(np, 0.5)
    assert res is not None
    res = core_math_ops._erfc(np, 0.5)
    assert res is not None
    res = core_math_ops._expm1(np, 0.5)
    assert res is not None

    # Mock fallback for erf, erfc, expm1, erfinv
    class DummyMock:
        @staticmethod
        def sign(x):
            return np.sign(x)

        @staticmethod
        def abs(x):
            return np.abs(x)

        @staticmethod
        def exp(x):
            return np.exp(x)

        @staticmethod
        def log(x):
            return np.log(x)

        @staticmethod
        def sqrt(x):
            return np.sqrt(x)

    dummy = DummyMock()
    res = core_math_ops._erf(dummy, 0.5)
    assert res is not None
    res = core_math_ops._erfc(dummy, 0.5)
    assert res is not None
    res = core_math_ops._expm1(dummy, 0.5)
    assert res is not None
    res = core_math_ops._erfinv(dummy, 0.5)
    assert res is not None

    # NanToNum
    res = core_math_ops._nan_to_num(np, np.array([np.nan, 1.0]))
    assert res is not None
    res = core_math_ops._nan_to_num(dummy, np.array([np.nan, 1.0]))
    assert res is None

    # Einsum
    res = core_math_ops._einsum(np, "i,i->", np.array([1.0]), np.array([2.0]))
    assert res is not None
    res = core_math_ops._einsum(dummy, "i,i->", np.array([1.0]), np.array([2.0]))
    assert res is None

    # Allclose
    res = core_math_ops._allclose(np, np.array([1.0]), np.array([1.0]))
    assert res is not None

    class DummyAllclose:
        class data:
            @staticmethod
            def item():
                return 1.0

    res = core_math_ops._allclose(dummy, DummyAllclose(), DummyAllclose())
    assert res is None

    # Psum, Pmean
    res = core_math_ops._psum(np, np.array([1.0]))
    assert res is not None
    res = core_math_ops._pmean(np, np.array([1.0]))
    assert res is not None

    # Polygamma
    class DummyMockPolygamma:
        pass

    res = core_math_ops._polygamma(DummyMockPolygamma, 1, 1)
    assert res is not None

    # Digamma
    class DummyMockDigamma:
        pass

    res = core_math_ops._digamma(DummyMockDigamma, 1)
    assert res is not None

    # Lgamma
    res = core_math_ops._mock_zerofraction(np, np.array([0.0, 1.0]))
    assert res is not None

    class MockRandomZeroFraction:
        class random:
            @staticmethod
            def zerofraction(*args, **kwargs):
                return 1.0

    res = core_math_ops._mock_zerofraction(MockRandomZeroFraction, np.array([0.0, 1.0]))
    assert res == 1.0

    # Zeta
    res = core_math_ops._mock_zeta(np, 2.0, 1.0)
    assert res is not None
    res = core_math_ops._mock_zeta(object, 2.0, 1.0)
    assert res is not None


def test_missing_i0():
    import numpy as np

    import ml_switcheroo_compiler.backends.eager.core_math_ops as mod

    class BkWithI0:
        def i0(self, x):
            return np.array([1.0])

        def iscomplex(self, x):
            return np.array([1.0])

        def isreal(self, x):
            return np.array([1.0])

        def signbit(self, x):
            return np.array([1.0])

    class BkWithoutI0:
        pass

    x = np.array([1.0])
    mod._i0(BkWithI0(), x)
    mod._i0(BkWithoutI0(), x)

    mod._iscomplex(BkWithI0(), x)
    mod._iscomplex(BkWithoutI0(), x)

    mod._isreal(BkWithI0(), x)
    mod._isreal(BkWithoutI0(), x)

    mod._signbit(BkWithI0(), x)
    mod._signbit(BkWithoutI0(), x)
