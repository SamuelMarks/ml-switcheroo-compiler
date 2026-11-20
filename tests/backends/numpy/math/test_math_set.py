"""Behavioral unit tests for math set."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import BackendArg, numpy_eager_registry
from tests.backends.numpy.eager._math_advanced_test_ns import math_set as math_set
from tests.testing_utils import MockTestStub, NumpyTestArray, require_callable


def test_math_set() -> None:
    """Test Union1d, Intersect1d, and Isin directly and via registry."""
    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array([2, 3, 4], dtype=np.int64)

    union = math_set._np_union1d(np, a, b)
    assert np.array_equal(union, np.array([1, 2, 3, 4]))

    reg_union = numpy_eager_registry.get("Union1d")
    assert np.array_equal(require_callable(reg_union)(np, a, b), np.array([1, 2, 3, 4]))

    class _SetStub(MockTestStub):
        """Stub backend module providing `intersect1d` and `isin`."""

        @staticmethod
        def intersect1d(*args: BackendArg, **kw: BackendArg) -> NumpyTestArray:
            """Return mock intersection array.

            Args:
                *args: Positional arguments.
                **kw: Keyword arguments.

            Returns:
                NumpyTestArray: Constant `[2, 3]` array.
            """
            del args, kw
            return np.array([2, 3])

        @staticmethod
        def isin(*args: BackendArg, **kw: BackendArg) -> NumpyTestArray:
            """Return mock containment boolean mask.

            Args:
                *args: Positional arguments.
                **kw: Keyword arguments.

            Returns:
                NumpyTestArray: Constant `[False, True, True]` mask.
            """
            del args, kw
            return np.array([False, True, True])

    mock_mod = _SetStub()

    inter = math_set._np_intersect1d_(mock_mod, a, b)
    assert np.array_equal(inter, np.array([2, 3]))

    reg_inter = numpy_eager_registry.get("Intersect1d")
    assert np.array_equal(require_callable(reg_inter)(mock_mod, a, b), np.array([2, 3]))

    isin = math_set._np_isin_(mock_mod, a, b)
    assert np.array_equal(isin, np.array([False, True, True]))

    reg_isin = numpy_eager_registry.get("Isin")
    assert np.array_equal(require_callable(reg_isin)(mock_mod, a, b), np.array([False, True, True]))
