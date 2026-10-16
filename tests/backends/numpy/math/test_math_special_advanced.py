"""Tests for test_math_special_advanced."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import math_special as special_mod

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_math_special_functions() -> None:
    """Test all special math operations in math_special.

    Returns:
        None
    """
    x = 0.5

    # BesselJ0 & BesselJ1
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_j0)(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_j0)(np, x=x))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_j0)(np), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_j0)(np, 0.0))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("BesselJ0")(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_j0)(np, x))

    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_j1)(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_j1)(np, x=x))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_j1)(np), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_j1)(np, 0.0))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("BesselJ1")(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_j1)(np, x))

    # BesselK0 & BesselK0e
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_k0)(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_k0)(np, x=x))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("BesselK0")(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_k0)(np, x))

    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_k0e)(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_k0e)(np, x=x))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("BesselK0e")(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_k0e)(np, x))

    # BesselK1 & BesselK1e
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_k1)(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_k1)(np, x=x))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("BesselK1")(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_k1)(np, x))

    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_k1e)(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_k1e)(np, x=x))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("BesselK1e")(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_k1e)(np, x))

    # BesselY0 & BesselY1
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_y0)(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_y0)(np, x=x))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("BesselY0")(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_y0)(np, x))

    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_y1)(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_y1)(np, x=x))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("BesselY1")(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_y1)(np, x))

    # Dawsn
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_dawsn)(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_dawsn)(np, x=x))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("Dawsn")(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_dawsn)(np, x))

    # Expint with args > 1, args == 1 with kwargs, and empty
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_expint)(np, x, 2), cast("Callable[..., EagerTensorProtocol]", special_mod._np_expint)(np, x, n=2))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_expint)(np, x=x, n=2), cast("Callable[..., EagerTensorProtocol]", special_mod._np_expint)(np, x, 2))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_expint)(np), cast("Callable[..., EagerTensorProtocol]", special_mod._np_expint)(np, 0.0, 1))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("Expint")(np, x, 2), cast("Callable[..., EagerTensorProtocol]", special_mod._np_expint)(np, x, 2))

    # FresnelCos & FresnelSin
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_fresnel_cos)(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_fresnel_cos)(np, x=x))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_fresnel_cos)(np), cast("Callable[..., EagerTensorProtocol]", special_mod._np_fresnel_cos)(np, 0.0))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("FresnelCos")(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_fresnel_cos)(np, x))

    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_fresnel_sin)(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_fresnel_sin)(np, x=x))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_fresnel_sin)(np), cast("Callable[..., EagerTensorProtocol]", special_mod._np_fresnel_sin)(np, 0.0))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("FresnelSin")(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_fresnel_sin)(np, x))

    # Spence
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_spence)(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_spence)(np, x=x))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_spence)(np), cast("Callable[..., EagerTensorProtocol]", special_mod._np_spence)(np, 0.0))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("Spence")(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_spence)(np, x))

    # BesselI0 & BesselI1 & BesselJn
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_i0)(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_i0)(np, x=x))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("BesselI0")(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_i0)(np, x))

    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_i1)(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_i1)(np, x=x))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("BesselI1")(np, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_i1)(np, x))

    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_jn)(np, 1.0, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_jn)(np, x=1.0, y=x))
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("BesselJn")(np, 1.0, x), cast("Callable[..., EagerTensorProtocol]", special_mod._np_bessel_jn)(np, 1.0, x))

    # Bartlett
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", special_mod._np_bartlett)(np, 5), cast("Callable[..., EagerTensorProtocol]", np.bartlett)(5))
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("Bartlett")(np, 5), cast("Callable[..., EagerTensorProtocol]", np.bartlett)(5))
