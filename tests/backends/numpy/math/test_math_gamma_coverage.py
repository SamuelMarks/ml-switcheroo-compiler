"""Tests for test_math_gamma_coverage."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_gamma as math_gamma

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_math_gamma() -> None:
    """Test multivariate log gamma evaluation directly and via registry."""
    mock_module = types.SimpleNamespace(mvlgamma=lambda a, p: np.array(42.0))
    res = math_gamma._np_mvlgamma(mock_module, np.array(2.5), 2)
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(res, 42.0)

    reg_mvl = numpy_eager_registry.get("Mvlgamma")
    res_reg = cast("Callable[..., EagerTensorProtocol]", reg_mvl)(mock_module, cast("Callable[..., EagerTensorProtocol]", np.array)(3.0), 3)
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(res_reg, 42.0)
