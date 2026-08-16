"""Tests for extra."""

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.math_unary import _np_erf, _np_erfc, _np_erfinv, _np_exp2, _np_igamma, _np_igammac, _np_signbit


class DummyModule:
    """Module without asarray."""

    def vectorize(self, f):
        """Vectorize."""

        def _f(x):
            """F."""
            return [f(i) for i in x]

        return _f


def test_erf_erfc_no_asarray() -> None:
    """Test coverage for branches."""
    mod = DummyModule()

    # x is a scalar
    res_erf = _np_erf(mod, 0.0)
    assert res_erf == 0.0

    res_erfc = _np_erfc(mod, 0.0)
    assert res_erfc == 1.0


def test_math_unary_missing():
    # test scalar
    assert _np_erf(np, 0.0) == 0.0
    assert _np_erfc(np, 0.0) == 1.0

    # test scipy functions
    assert _np_erfinv(np, 0.0) == 0.0
    assert _np_igamma(np, 1.0, 1.0) is not None
    assert _np_igammac(np, 1.0, 1.0) is not None

    # kwargs
    assert _np_igamma(np, 1.0, x=1.0) is not None
    assert _np_igammac(np, 1.0, x=1.0) is not None

    # exp2 and signbit
    assert _np_exp2(np, 2.0) == 4.0
    assert _np_signbit(np, -1.0) == True

    # array inputs to hit vectorize
    assert _np_erf(np, np.array([0.0])) is not None
    assert _np_erfc(np, np.array([0.0])) is not None

    from ml_switcheroo_compiler.backends.numpy.eager.math_unary import _np_angle, _np_bitwise_not, _np_exp, _np_expm1, _np_isfinite, _np_isinf, _np_isnan, _np_log, _np_log1p, _np_log2, _np_log10, _np_round

    assert _np_exp(np, 0.0) == 1.0
    assert _np_log(np, 1.0) == 0.0
    assert _np_log1p(np, 0.0) == 0.0
    assert _np_round(np, 0.0) == 0.0
    assert _np_angle(np, 1.0) == 0.0
    assert _np_expm1(np, 0.0) == 0.0
    assert _np_log10(np, 1.0) == 0.0
    assert _np_log2(np, 1.0) == 0.0
    assert _np_isinf(np, 0.0) == False
    assert _np_isfinite(np, 0.0) == True
    assert _np_bitwise_not(np, np.array([1])) is not None
    assert _np_isnan(np, np.array([0.0])) is not None
