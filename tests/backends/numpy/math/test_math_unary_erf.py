"""Tests for extra."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.math_unary import _np_erf, _np_erfc, _np_erfinv, _np_exp2, _np_igamma, _np_igammac, _np_signbit
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mod

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


class DummyModule:
    """Module without asarray."""

    def vectorize(self, f: EagerValue) -> EagerValue:
        """Vectorize.

        Args:
            f: Test parameter `f`.
        """

        def _f(x: EagerValue) -> EagerValue:
            """F.

            Args:
                x: Test parameter `x`.
            """
            return [ensure_callable(f)(i) for i in ensure_eager_tensor(x)]

        return _f


def test_erf_erfc_no_asarray() -> None:
    """Test coverage for branches."""
    mod = DummyModule()

    # x is a scalar
    res_erf = _np_erf(ensure_mod(mod), 0.0)
    assert res_erf == 0.0

    res_erfc = _np_erfc(ensure_mod(mod), 0.0)
    assert res_erfc == 1.0


def test_math_unary_missing() -> None:
    # test scalar
    """Test function test_math_unary_missing."""
    assert _np_erf(ensure_eager_tensor(np), 0.0) == 0.0
    assert _np_erfc(ensure_eager_tensor(np), 0.0) == 1.0

    # test scipy functions
    assert _np_erfinv(ensure_eager_tensor(np), 0.0) == 0.0
    assert _np_igamma(ensure_eager_tensor(np), 1.0, 1.0) is not None
    assert _np_igammac(ensure_eager_tensor(np), 1.0, 1.0) is not None

    # kwargs
    assert _np_igamma(ensure_eager_tensor(np), 1.0, x=1.0) is not None
    assert _np_igammac(ensure_eager_tensor(np), 1.0, x=1.0) is not None

    # exp2 and signbit
    assert _np_exp2(ensure_eager_tensor(np), 2.0) == 4.0
    assert _np_signbit(ensure_eager_tensor(np), -1.0)

    # array inputs to hit vectorize
    assert _np_erf(ensure_eager_tensor(np), np.array([0.0])) is not None
    assert _np_erfc(ensure_eager_tensor(np), np.array([0.0])) is not None

    from ml_switcheroo_compiler.backends.numpy.eager.math_unary import _np_angle, _np_bitwise_not, _np_exp, _np_expm1, _np_isfinite, _np_isinf, _np_isnan, _np_log, _np_log1p, _np_log2, _np_log10, _np_round

    assert _np_exp(ensure_eager_tensor(np), 0.0) == 1.0
    assert _np_log(ensure_eager_tensor(np), 1.0) == 0.0
    assert _np_log1p(ensure_eager_tensor(np), 0.0) == 0.0
    assert _np_round(ensure_eager_tensor(np), 0.0) == 0.0
    assert _np_angle(ensure_eager_tensor(np), 1.0) == 0.0
    assert _np_expm1(ensure_eager_tensor(np), 0.0) == 0.0
    assert _np_log10(ensure_eager_tensor(np), 1.0) == 0.0
    assert _np_log2(ensure_eager_tensor(np), 1.0) == 0.0
    assert not _np_isinf(ensure_eager_tensor(np), 0.0)
    assert _np_isfinite(ensure_eager_tensor(np), 0.0)
    assert _np_bitwise_not(ensure_eager_tensor(np), np.array([1])) is not None
    assert _np_isnan(ensure_eager_tensor(np), np.array([0.0])) is not None
