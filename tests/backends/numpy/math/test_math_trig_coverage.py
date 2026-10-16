"""Tests for test_math_trig_coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import math_trig as trig_module

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_math_trig_ops() -> None:
    """Test trigonometric and degree operations directly and through eager registry.

    Returns:
        None
    """
    val = np.array([0.0, 0.5])

    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", trig_module._np_sin)(np, val), cast("Callable[..., EagerTensorProtocol]", np.sin)(val))
    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("Sin")(np, val), cast("Callable[..., EagerTensorProtocol]", np.sin)(val))

    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", trig_module._np_cos)(np, val), cast("Callable[..., EagerTensorProtocol]", np.cos)(val))
    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("Cos")(np, val), cast("Callable[..., EagerTensorProtocol]", np.cos)(val))

    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", trig_module._np_acos)(np, val), cast("Callable[..., EagerTensorProtocol]", np.arccos)(val))
    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("Acos")(np, val), cast("Callable[..., EagerTensorProtocol]", np.arccos)(val))

    val_cosh = np.array([1.5, 2.0])
    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", trig_module._np_acosh)(np, val_cosh), cast("Callable[..., EagerTensorProtocol]", np.arccosh)(val_cosh))
    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("Acosh")(np, val_cosh), cast("Callable[..., EagerTensorProtocol]", np.arccosh)(val_cosh))

    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", trig_module._np_asin)(np, val), cast("Callable[..., EagerTensorProtocol]", np.arcsin)(val))
    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("Asin")(np, val), cast("Callable[..., EagerTensorProtocol]", np.arcsin)(val))

    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", trig_module._np_asinh)(np, val), cast("Callable[..., EagerTensorProtocol]", np.arcsinh)(val))
    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("Asinh")(np, val), cast("Callable[..., EagerTensorProtocol]", np.arcsinh)(val))

    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", trig_module._np_atan)(np, val), cast("Callable[..., EagerTensorProtocol]", np.arctan)(val))
    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("Atan")(np, val), cast("Callable[..., EagerTensorProtocol]", np.arctan)(val))

    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", trig_module._np_atanh)(np, val), cast("Callable[..., EagerTensorProtocol]", np.arctanh)(val))
    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("Atanh")(np, val), cast("Callable[..., EagerTensorProtocol]", np.arctanh)(val))

    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", trig_module._np_atan2)(np, val, val + 1.0), cast("Callable[..., EagerTensorProtocol]", np.arctan2)(val, val + 1.0))
    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("Atan2")(np, val, val + 1.0), cast("Callable[..., EagerTensorProtocol]", np.arctan2)(val, val + 1.0))

    deg = np.array([0.0, 90.0, 180.0])
    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", trig_module._np_deg2rad)(np, deg), cast("Callable[..., EagerTensorProtocol]", np.deg2rad)(deg))
    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("Deg2Rad")(np, deg), cast("Callable[..., EagerTensorProtocol]", np.deg2rad)(deg))

    rad = np.array([0.0, np.pi / 2.0, np.pi])
    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", trig_module._np_rad2deg)(np, rad), cast("Callable[..., EagerTensorProtocol]", np.rad2deg)(rad))
    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)("Rad2Deg")(np, rad), cast("Callable[..., EagerTensorProtocol]", np.rad2deg)(rad))
