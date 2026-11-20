"""Behavioral unit tests for math gamma."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import BackendArg, numpy_eager_registry
from tests.backends.numpy.eager._math_advanced_test_ns import math_gamma as math_gamma
from tests.testing_utils import MockTestStub, NumpyTestArray, require_callable


def test_math_gamma() -> None:
    """Test multivariate log gamma evaluation directly and via registry."""

    class _GammaStub(MockTestStub):
        """Stub backend module providing `mvlgamma`."""

        @staticmethod
        def mvlgamma(a: BackendArg, p: int) -> NumpyTestArray:
            """Evaluate mock multivariate log-gamma.

            Args:
                a: Input array.
                p: Dimension parameter.

            Returns:
                NumpyTestArray: Constant scalar array `42.0`.
            """
            del a, p
            return np.array(42.0)

    mock_module = _GammaStub()
    res = math_gamma._np_mvlgamma(mock_module, np.array(2.5), 2)
    assert np.isclose(res, 42.0)

    reg_mvl = numpy_eager_registry.get("Mvlgamma")
    res_reg = require_callable(reg_mvl)(mock_module, np.array(3.0), 3)
    assert np.isclose(res_reg, 42.0)
