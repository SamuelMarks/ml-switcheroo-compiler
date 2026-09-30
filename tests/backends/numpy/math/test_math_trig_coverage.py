"""Tests for test_math_trig_coverage."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import math_trig as trig_module


def test_math_trig_ops() -> None:
    """Test trigonometric and degree operations directly and through eager registry.

    Returns:
        None
    """
    val = np.array([0.0, 0.5])

    assert np.allclose(trig_module._np_sin(np, val), np.sin(val))
    assert np.allclose(numpy_eager_registry.get("Sin")(np, val), np.sin(val))

    assert np.allclose(trig_module._np_cos(np, val), np.cos(val))
    assert np.allclose(numpy_eager_registry.get("Cos")(np, val), np.cos(val))

    assert np.allclose(trig_module._np_acos(np, val), np.arccos(val))
    assert np.allclose(numpy_eager_registry.get("Acos")(np, val), np.arccos(val))

    val_cosh = np.array([1.5, 2.0])
    assert np.allclose(trig_module._np_acosh(np, val_cosh), np.arccosh(val_cosh))
    assert np.allclose(numpy_eager_registry.get("Acosh")(np, val_cosh), np.arccosh(val_cosh))

    assert np.allclose(trig_module._np_asin(np, val), np.arcsin(val))
    assert np.allclose(numpy_eager_registry.get("Asin")(np, val), np.arcsin(val))

    assert np.allclose(trig_module._np_asinh(np, val), np.arcsinh(val))
    assert np.allclose(numpy_eager_registry.get("Asinh")(np, val), np.arcsinh(val))

    assert np.allclose(trig_module._np_atan(np, val), np.arctan(val))
    assert np.allclose(numpy_eager_registry.get("Atan")(np, val), np.arctan(val))

    assert np.allclose(trig_module._np_atanh(np, val), np.arctanh(val))
    assert np.allclose(numpy_eager_registry.get("Atanh")(np, val), np.arctanh(val))

    assert np.allclose(trig_module._np_atan2(np, val, val + 1.0), np.arctan2(val, val + 1.0))
    assert np.allclose(numpy_eager_registry.get("Atan2")(np, val, val + 1.0), np.arctan2(val, val + 1.0))

    deg = np.array([0.0, 90.0, 180.0])
    assert np.allclose(trig_module._np_deg2rad(np, deg), np.deg2rad(deg))
    assert np.allclose(numpy_eager_registry.get("Deg2Rad")(np, deg), np.deg2rad(deg))

    rad = np.array([0.0, np.pi / 2.0, np.pi])
    assert np.allclose(trig_module._np_rad2deg(np, rad), np.rad2deg(rad))
    assert np.allclose(numpy_eager_registry.get("Rad2Deg")(np, rad), np.rad2deg(rad))
