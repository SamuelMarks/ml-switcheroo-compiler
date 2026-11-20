"""Tests for test_core_math_special."""

from __future__ import annotations

import math
import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_special import (
    _bessel_i0,
    _bessel_i0e,
    _bessel_i1,
    _bessel_i1e,
    _bessel_j0,
    _bessel_j1,
    _bessel_jn,
    _bessel_k0,
    _bessel_k0e,
    _bessel_k1,
    _bessel_k1e,
    _bessel_y0,
    _bessel_y1,
    _digamma,
    _erf,
    _erfc,
    _erfinv,
    _gamma,
    _igamma,
    _igammac,
    _np_modifiedbesseli1,
    _polygamma,
)
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import MockTestStub, extract_test_shape

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class _EmptyBackend(MockTestStub):
    """Mock backend without any attributes."""

    pass


def test_math_special_all_branches() -> None:
    """Test 100% lines and branches of math_special.py.

    Returns:
        None
    """
    empty_backend = _EmptyBackend()

    # 1. _erf
    # Case A: backend has erf
    class BackendErf(MockTestStub):
        """Backend with erf."""

        def erf(self, x: float) -> float:
            """Erf.

            Args:
                x: Test parameter `x`.

            Returns:
                float: Computed output value from `erf`.
            """
            return math.erf(x)

    assert math.isclose(ensure_ndarray(_erf(BackendErf(), 0.5)), math.erf(0.5))

    # Case B: backend lacks erf -> approximation branch
    class BackendMathOps(MockTestStub):
        """Backend with sign, abs, exp, log, sqrt."""

        def sign(self, x: BackendArg) -> BackendArg:
            """Sign.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `sign`.
            """
            return np.sign(ensure_ndarray(x))

        def abs(self, x: BackendArg) -> BackendArg:
            """Abs.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `abs`.
            """
            return np.abs(ensure_ndarray(x))

        def exp(self, x: BackendArg) -> BackendArg:
            """Exp.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `exp`.
            """
            return np.exp(ensure_ndarray(x))

        def log(self, x: BackendArg) -> BackendArg:
            """Log.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `log`.
            """
            return np.log(ensure_ndarray(x))

        def sqrt(self, x: BackendArg) -> BackendArg:
            """Sqrt.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `sqrt`.
            """
            return np.sqrt(ensure_ndarray(x))

    bmo = BackendMathOps()
    erf_approx = _erf(bmo, np.array([-0.5, 0.0, 0.5]))
    assert np.allclose(
        ensure_ndarray(erf_approx),
        [math.erf(-0.5), math.erf(0.0), math.erf(0.5)],
        atol=1e-4,
    )

    # 2. _erfc
    # Case A: backend has erfc
    class BackendErfc(MockTestStub):
        """Backend with erfc."""

        def erfc(self, x: float) -> float:
            """Erfc.

            Args:
                x: Test parameter `x`.

            Returns:
                float: Computed output value from `erfc`.
            """
            return math.erfc(x)

    assert math.isclose(ensure_ndarray(_erfc(BackendErfc(), 0.5)), math.erfc(0.5))

    # Case B: backend lacks erfc -> fallback to 1.0 - _erf
    erfc_fallback = _erfc(bmo, 0.5)
    assert np.isclose(ensure_ndarray(erfc_fallback), math.erfc(0.5), atol=1e-4)

    # 3. _erfinv
    # Case A: backend has erfinv
    class BackendErfinv(MockTestStub):
        """Backend with erfinv."""

        def erfinv(self, x: float) -> str:
            """Erfinv.

            Args:
                x: Test parameter `x`.

            Returns:
                str: Computed output value from `erfinv`.
            """
            return "erfinv_result"

    assert _erfinv(BackendErfinv(), 0.5) == "erfinv_result"

    # Case B: backend lacks erfinv -> approximation
    erfinv_approx = _erfinv(bmo, np.array([0.5]))
    assert extract_test_shape(erfinv_approx) == (1,)
    assert ensure_ndarray(erfinv_approx)[0] > 0

    # 4. SciPy special functions
    x_val = 1.5
    assert np.isclose(ensure_ndarray(_bessel_i0(empty_backend, x_val)), 1.64672, atol=1e-3)
    assert np.isclose(ensure_ndarray(_bessel_i0e(empty_backend, x_val)), 0.36743, atol=1e-3)
    assert np.isclose(ensure_ndarray(_bessel_i1(empty_backend, x_val)), 0.981666, atol=1e-3)
    assert np.isclose(ensure_ndarray(_bessel_i1e(empty_backend, x_val)), 0.219039, atol=1e-3)
    assert np.isclose(ensure_ndarray(_bessel_j0(empty_backend, x_val)), 0.511827, atol=1e-3)
    assert np.isclose(ensure_ndarray(_bessel_j1(empty_backend, x_val)), 0.557936, atol=1e-3)
    assert np.isclose(ensure_ndarray(_bessel_jn(empty_backend, 2, x_val)), 0.232087, atol=1e-3)
    assert np.isclose(ensure_ndarray(_bessel_k0(empty_backend, x_val)), 0.213805, atol=1e-3)
    assert np.isclose(ensure_ndarray(_bessel_k0e(empty_backend, x_val)), 0.958210, atol=1e-3)
    assert np.isclose(ensure_ndarray(_bessel_k1(empty_backend, x_val)), 0.277387, atol=1e-3)
    assert np.isclose(ensure_ndarray(_bessel_k1e(empty_backend, x_val)), 1.243165, atol=1e-3)
    assert np.isclose(ensure_ndarray(_bessel_y0(empty_backend, x_val)), 0.382448, atol=1e-3)
    assert np.isclose(ensure_ndarray(_bessel_y1(empty_backend, x_val)), -0.412308, atol=1e-3)

    assert np.isclose(ensure_ndarray(_digamma(empty_backend, x_val)), 0.036489, atol=1e-3)
    assert np.isclose(ensure_ndarray(_igammac(empty_backend, 1.0, x_val)), 0.223130, atol=1e-3)
    assert np.isclose(ensure_ndarray(_polygamma(empty_backend, 1, x_val)), 0.934802, atol=1e-3)
    assert np.isclose(ensure_ndarray(_igamma(empty_backend, 1.0, x_val)), 0.776869, atol=1e-3)
    assert np.isclose(ensure_ndarray(_gamma(empty_backend, x_val)), 0.886226, atol=1e-3)
    assert np.isclose(ensure_ndarray(_np_modifiedbesseli1(empty_backend, x_val)), 0.981666, atol=1e-3)

    # Registry verification
    for op_name in [
        "Erf",
        "Erfc",
        "Erfinv",
        "BesselI0",
        "BesselI0e",
        "BesselI1",
        "BesselI1e",
        "BesselJ0",
        "BesselJ1",
        "BesselJn",
        "BesselK0",
        "BesselK0e",
        "BesselK1",
        "BesselK1e",
        "BesselY0",
        "BesselY1",
        "Digamma",
        "Igammac",
        "Polygamma",
        "Igamma",
        "Gamma",
        "ModifiedBesselI1",
    ]:
        assert global_eager_registry.get(op_name) is not None
