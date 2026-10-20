"""Tests for test_math_gamma_coverage."""

from __future__ import annotations

import types
import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_gamma as math_gamma
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_math_gamma() -> None:
    """Test multivariate log gamma evaluation directly and via registry."""
    mock_module = types.SimpleNamespace(mvlgamma=lambda a, p: np.array(42.0))
    res = math_gamma._np_mvlgamma(typing.cast("BackendModuleProtocol", mock_module), np.array(2.5), 2)
    assert typing.cast("EagerCallableProtocol", np.isclose)(res, 42.0)

    reg_mvl = numpy_eager_registry.get("Mvlgamma")
    res_reg = typing.cast("EagerCallableProtocol", reg_mvl)(mock_module, typing.cast("EagerCallableProtocol", np.array)(3.0), 3)
    assert typing.cast("EagerCallableProtocol", np.isclose)(res_reg, 42.0)
