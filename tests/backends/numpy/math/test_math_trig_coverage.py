"""Tests for test_math_trig_coverage."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import math_trig as trig_module
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    pass


def test_math_trig_ops() -> None:
    """Test trigonometric and degree operations directly and through eager registry.

    Returns:
        None
    """
    val = np.array([0.0, 0.5])

    assert as_callable(np.allclose)(as_callable(trig_module._np_sin)(np, val), as_callable(np.sin)(val))
    assert as_callable(np.allclose)(as_callable(numpy_eager_registry.get)("Sin")(np, val), as_callable(np.sin)(val))

    assert as_callable(np.allclose)(as_callable(trig_module._np_cos)(np, val), as_callable(np.cos)(val))
    assert as_callable(np.allclose)(as_callable(numpy_eager_registry.get)("Cos")(np, val), as_callable(np.cos)(val))

    assert as_callable(np.allclose)(as_callable(trig_module._np_acos)(np, val), as_callable(np.arccos)(val))
    assert as_callable(np.allclose)(as_callable(numpy_eager_registry.get)("Acos")(np, val), as_callable(np.arccos)(val))

    val_cosh = np.array([1.5, 2.0])
    assert as_callable(np.allclose)(as_callable(trig_module._np_acosh)(np, val_cosh), as_callable(np.arccosh)(val_cosh))
    assert as_callable(np.allclose)(as_callable(numpy_eager_registry.get)("Acosh")(np, val_cosh), as_callable(np.arccosh)(val_cosh))

    assert as_callable(np.allclose)(as_callable(trig_module._np_asin)(np, val), as_callable(np.arcsin)(val))
    assert as_callable(np.allclose)(as_callable(numpy_eager_registry.get)("Asin")(np, val), as_callable(np.arcsin)(val))

    assert as_callable(np.allclose)(as_callable(trig_module._np_asinh)(np, val), as_callable(np.arcsinh)(val))
    assert as_callable(np.allclose)(as_callable(numpy_eager_registry.get)("Asinh")(np, val), as_callable(np.arcsinh)(val))

    assert as_callable(np.allclose)(as_callable(trig_module._np_atan)(np, val), as_callable(np.arctan)(val))
    assert as_callable(np.allclose)(as_callable(numpy_eager_registry.get)("Atan")(np, val), as_callable(np.arctan)(val))

    assert as_callable(np.allclose)(as_callable(trig_module._np_atanh)(np, val), as_callable(np.arctanh)(val))
    assert as_callable(np.allclose)(as_callable(numpy_eager_registry.get)("Atanh")(np, val), as_callable(np.arctanh)(val))

    assert as_callable(np.allclose)(as_callable(trig_module._np_atan2)(np, val, val + 1.0), as_callable(np.arctan2)(val, val + 1.0))
    assert as_callable(np.allclose)(as_callable(numpy_eager_registry.get)("Atan2")(np, val, val + 1.0), as_callable(np.arctan2)(val, val + 1.0))

    deg = np.array([0.0, 90.0, 180.0])
    assert as_callable(np.allclose)(as_callable(trig_module._np_deg2rad)(np, deg), as_callable(np.deg2rad)(deg))
    assert as_callable(np.allclose)(as_callable(numpy_eager_registry.get)("Deg2Rad")(np, deg), as_callable(np.deg2rad)(deg))

    rad = np.array([0.0, np.pi / 2.0, np.pi])
    assert as_callable(np.allclose)(as_callable(trig_module._np_rad2deg)(np, rad), as_callable(np.rad2deg)(rad))
    assert as_callable(np.allclose)(as_callable(numpy_eager_registry.get)("Rad2Deg")(np, rad), as_callable(np.rad2deg)(rad))
