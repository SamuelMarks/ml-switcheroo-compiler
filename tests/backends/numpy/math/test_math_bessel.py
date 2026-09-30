"""Tests for test_math_bessel."""

from __future__ import annotations

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_bessel as bessel_mod


def test_math_bessel_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test all Bessel functions in math_bessel including fallback branches.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for monkeypatching.

    Returns:
        None
    """
    arr = np.array([0.5, 1.0])

    # Direct and registered BesselI0e and BesselI1e
    assert np.all(bessel_mod._np_bessel_i0e(np, arr) > 0)
    assert np.all(numpy_eager_registry.get("BesselI0e")(np, arr) > 0)
    assert np.all(bessel_mod._np_bessel_i1e(np, arr) > 0)
    assert np.all(numpy_eager_registry.get("BesselI1e")(np, arr) > 0)

    # modified_bessel_i0
    assert bessel_mod._np_modified_bessel_i0(np) is None
    assert np.allclose(bessel_mod._np_modified_bessel_i0(np, arr), np.i0(arr))
    assert np.allclose(numpy_eager_registry.get("modified_bessel_i0")(np, arr), np.i0(arr))

    # modified_bessel_i1 with scipy
    assert bessel_mod._np_modified_bessel_i1(np) is None
    assert np.all(bessel_mod._np_modified_bessel_i1(np, arr) > 0)
    assert np.all(numpy_eager_registry.get("modified_bessel_i1")(np, arr) > 0)

    # modified_bessel_k0 with scipy
    assert bessel_mod._np_modified_bessel_k0(np) is None
    assert np.all(bessel_mod._np_modified_bessel_k0(np, arr) > 0)
    assert np.all(numpy_eager_registry.get("modified_bessel_k0")(np, arr) > 0)

    # modified_bessel_k1 with scipy
    assert bessel_mod._np_modified_bessel_k1(np) is None
    assert np.all(bessel_mod._np_modified_bessel_k1(np, arr) > 0)
    assert np.all(numpy_eager_registry.get("modified_bessel_k1")(np, arr) > 0)

    # Test fallback branch when scipy is None
    monkeypatch.setattr(bessel_mod, "_get_sc", lambda: None)
    # Array inputs (ndim > 0)
    i1_fallback_arr = bessel_mod._np_modified_bessel_i1(np, arr)
    assert i1_fallback_arr is not None and i1_fallback_arr.shape == (2,)

    k0_fallback_arr = bessel_mod._np_modified_bessel_k0(np, arr)
    assert k0_fallback_arr is not None and k0_fallback_arr.shape == (2,)

    k1_fallback_arr = bessel_mod._np_modified_bessel_k1(np, arr)
    assert k1_fallback_arr is not None and k1_fallback_arr.shape == (2,)

    # Scalar inputs (ndim == 0)
    scalar_val = np.array(1.0)
    i1_fallback_scalar = bessel_mod._np_modified_bessel_i1(np, scalar_val)
    assert i1_fallback_scalar is not None and i1_fallback_scalar.shape == ()

    k0_fallback_scalar = bessel_mod._np_modified_bessel_k0(np, scalar_val)
    assert k0_fallback_scalar is not None and k0_fallback_scalar.shape == ()

    k1_fallback_scalar = bessel_mod._np_modified_bessel_k1(np, scalar_val)
    assert k1_fallback_scalar is not None and k1_fallback_scalar.shape == ()
