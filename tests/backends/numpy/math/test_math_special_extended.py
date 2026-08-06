import ml_switcheroo_compiler.ops as _ops
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import (
    _np_frombuffer,
    _np_shifted_chebyshev_polynomial_t,
    _np_shifted_chebyshev_polynomial_u,
    _np_shifted_chebyshev_polynomial_v,
    _np_shifted_chebyshev_polynomial_w,
)


class MockBackend:
    def descriptive(self, *args, **kwargs):
        return "desc"

    def distributions(self, *args, **kwargs):
        return "dist"


def test_descriptive_distributions_fallback():
    # Get the lowercase ones from registry, which point to the first definitions
    _np_descriptive_first = numpy_eager_registry._registry["descriptive"]
    _np_distributions_first = numpy_eager_registry._registry["distributions"]

    res_desc = _np_descriptive_first(MockBackend())
    assert res_desc == "desc"

    res_dist = _np_distributions_first(MockBackend())
    assert res_dist == "dist"

    # Cover lines returning numpy implementations
    class EmptyBackend:
        pass

    assert _np_descriptive_first(EmptyBackend()).shape == (3,)
    assert _np_distributions_first(EmptyBackend()).shape == (2,)

    class DummyType:
        def __init__(self, *args, **kwargs):
            self.res = "dummy_type"

        def __new__(cls, *args, **kwargs):
            return "dummy_type"

    _ops.descriptive = DummyType
    _ops.distributions = DummyType
    try:
        res_desc = _np_descriptive_first(None)
        res_dist = _np_distributions_first(None)
        if hasattr(res_desc, "shape"):
            pass
        else:
            assert res_desc == "dummy_type"
            assert res_dist == "dummy_type"
    finally:
        del _ops.descriptive
        del _ops.distributions

    # Cover the except Exception block
    class ErrorType:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Boom")

        def __new__(cls, *args, **kwargs):
            raise RuntimeError("Boom")

    _ops.descriptive = ErrorType
    _ops.distributions = ErrorType
    try:
        # Should catch exception and fall back to zeros
        assert _np_descriptive_first(EmptyBackend()).shape == (3,)
        assert _np_distributions_first(EmptyBackend()).shape == (2,)
    finally:
        del _ops.descriptive
        del _ops.distributions


def test_shifted_chebyshev_missing_args():
    assert _np_shifted_chebyshev_polynomial_t(None, x=None) is None
    assert _np_shifted_chebyshev_polynomial_u(None, x=None) is None
    assert _np_shifted_chebyshev_polynomial_v(None, x=None) is None
    assert _np_shifted_chebyshev_polynomial_w(None, x=None) is None


def test_frombuffer_none():
    assert _np_frombuffer(None) is None
    buf = b"12345678"
    assert _np_frombuffer(None, buf).shape == (1,)
