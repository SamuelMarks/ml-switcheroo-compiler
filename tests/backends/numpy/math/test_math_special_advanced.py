"""Tests for test_math_special_advanced."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import math_special as special_mod
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    pass


def test_math_special_functions() -> None:
    """Test all special math operations in math_special.

    Returns:
        None
    """
    x = 0.5

    # BesselJ0 & BesselJ1
    assert as_callable(np.isclose)(as_callable(special_mod._np_bessel_j0)(np, x), as_callable(special_mod._np_bessel_j0)(np, x=x))
    assert as_callable(np.isclose)(as_callable(special_mod._np_bessel_j0)(np), as_callable(special_mod._np_bessel_j0)(np, 0.0))
    assert as_callable(np.isclose)(as_callable(numpy_eager_registry.get)("BesselJ0")(np, x), as_callable(special_mod._np_bessel_j0)(np, x))

    assert as_callable(np.isclose)(as_callable(special_mod._np_bessel_j1)(np, x), as_callable(special_mod._np_bessel_j1)(np, x=x))
    assert as_callable(np.isclose)(as_callable(special_mod._np_bessel_j1)(np), as_callable(special_mod._np_bessel_j1)(np, 0.0))
    assert as_callable(np.isclose)(as_callable(numpy_eager_registry.get)("BesselJ1")(np, x), as_callable(special_mod._np_bessel_j1)(np, x))

    # BesselK0 & BesselK0e
    assert as_callable(np.isclose)(as_callable(special_mod._np_bessel_k0)(np, x), as_callable(special_mod._np_bessel_k0)(np, x=x))
    assert as_callable(np.isclose)(as_callable(numpy_eager_registry.get)("BesselK0")(np, x), as_callable(special_mod._np_bessel_k0)(np, x))

    assert as_callable(np.isclose)(as_callable(special_mod._np_bessel_k0e)(np, x), as_callable(special_mod._np_bessel_k0e)(np, x=x))
    assert as_callable(np.isclose)(as_callable(numpy_eager_registry.get)("BesselK0e")(np, x), as_callable(special_mod._np_bessel_k0e)(np, x))

    # BesselK1 & BesselK1e
    assert as_callable(np.isclose)(as_callable(special_mod._np_bessel_k1)(np, x), as_callable(special_mod._np_bessel_k1)(np, x=x))
    assert as_callable(np.isclose)(as_callable(numpy_eager_registry.get)("BesselK1")(np, x), as_callable(special_mod._np_bessel_k1)(np, x))

    assert as_callable(np.isclose)(as_callable(special_mod._np_bessel_k1e)(np, x), as_callable(special_mod._np_bessel_k1e)(np, x=x))
    assert as_callable(np.isclose)(as_callable(numpy_eager_registry.get)("BesselK1e")(np, x), as_callable(special_mod._np_bessel_k1e)(np, x))

    # BesselY0 & BesselY1
    assert as_callable(np.isclose)(as_callable(special_mod._np_bessel_y0)(np, x), as_callable(special_mod._np_bessel_y0)(np, x=x))
    assert as_callable(np.isclose)(as_callable(numpy_eager_registry.get)("BesselY0")(np, x), as_callable(special_mod._np_bessel_y0)(np, x))

    assert as_callable(np.isclose)(as_callable(special_mod._np_bessel_y1)(np, x), as_callable(special_mod._np_bessel_y1)(np, x=x))
    assert as_callable(np.isclose)(as_callable(numpy_eager_registry.get)("BesselY1")(np, x), as_callable(special_mod._np_bessel_y1)(np, x))

    # Dawsn
    assert as_callable(np.isclose)(as_callable(special_mod._np_dawsn)(np, x), as_callable(special_mod._np_dawsn)(np, x=x))
    assert as_callable(np.isclose)(as_callable(numpy_eager_registry.get)("Dawsn")(np, x), as_callable(special_mod._np_dawsn)(np, x))

    # Expint with args > 1, args == 1 with kwargs, and empty
    assert as_callable(np.isclose)(as_callable(special_mod._np_expint)(np, x, 2), as_callable(special_mod._np_expint)(np, x, n=2))
    assert as_callable(np.isclose)(as_callable(special_mod._np_expint)(np, x=x, n=2), as_callable(special_mod._np_expint)(np, x, 2))
    assert as_callable(np.isclose)(as_callable(special_mod._np_expint)(np), as_callable(special_mod._np_expint)(np, 0.0, 1))
    assert as_callable(np.isclose)(as_callable(numpy_eager_registry.get)("Expint")(np, x, 2), as_callable(special_mod._np_expint)(np, x, 2))

    # FresnelCos & FresnelSin
    assert as_callable(np.isclose)(as_callable(special_mod._np_fresnel_cos)(np, x), as_callable(special_mod._np_fresnel_cos)(np, x=x))
    assert as_callable(np.isclose)(as_callable(special_mod._np_fresnel_cos)(np), as_callable(special_mod._np_fresnel_cos)(np, 0.0))
    assert as_callable(np.isclose)(as_callable(numpy_eager_registry.get)("FresnelCos")(np, x), as_callable(special_mod._np_fresnel_cos)(np, x))

    assert as_callable(np.isclose)(as_callable(special_mod._np_fresnel_sin)(np, x), as_callable(special_mod._np_fresnel_sin)(np, x=x))
    assert as_callable(np.isclose)(as_callable(special_mod._np_fresnel_sin)(np), as_callable(special_mod._np_fresnel_sin)(np, 0.0))
    assert as_callable(np.isclose)(as_callable(numpy_eager_registry.get)("FresnelSin")(np, x), as_callable(special_mod._np_fresnel_sin)(np, x))

    # Spence
    assert as_callable(np.isclose)(as_callable(special_mod._np_spence)(np, x), as_callable(special_mod._np_spence)(np, x=x))
    assert as_callable(np.isclose)(as_callable(special_mod._np_spence)(np), as_callable(special_mod._np_spence)(np, 0.0))
    assert as_callable(np.isclose)(as_callable(numpy_eager_registry.get)("Spence")(np, x), as_callable(special_mod._np_spence)(np, x))

    # BesselI0 & BesselI1 & BesselJn
    assert as_callable(np.isclose)(as_callable(special_mod._np_bessel_i0)(np, x), as_callable(special_mod._np_bessel_i0)(np, x=x))
    assert as_callable(np.isclose)(as_callable(numpy_eager_registry.get)("BesselI0")(np, x), as_callable(special_mod._np_bessel_i0)(np, x))

    assert as_callable(np.isclose)(as_callable(special_mod._np_bessel_i1)(np, x), as_callable(special_mod._np_bessel_i1)(np, x=x))
    assert as_callable(np.isclose)(as_callable(numpy_eager_registry.get)("BesselI1")(np, x), as_callable(special_mod._np_bessel_i1)(np, x))

    assert as_callable(np.isclose)(as_callable(special_mod._np_bessel_jn)(np, 1.0, x), as_callable(special_mod._np_bessel_jn)(np, x=1.0, y=x))
    assert as_callable(np.isclose)(as_callable(numpy_eager_registry.get)("BesselJn")(np, 1.0, x), as_callable(special_mod._np_bessel_jn)(np, 1.0, x))

    # Bartlett
    assert as_callable(np.array_equal)(as_callable(special_mod._np_bartlett)(np, 5), as_callable(np.bartlett)(5))
    assert as_callable(np.array_equal)(as_callable(numpy_eager_registry.get)("Bartlett")(np, 5), as_callable(np.bartlett)(5))
