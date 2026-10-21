"""Tests for test_math_set_coverage."""

from __future__ import annotations

import types
import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_set as math_set
from tests.testing_utils import as_callable, as_mod

if typing.TYPE_CHECKING:
    pass


def test_math_set() -> None:
    """Test Union1d, Intersect1d, and Isin directly and via registry."""
    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array([2, 3, 4], dtype=np.int64)

    union = math_set._np_union1d(np, a, b)
    assert as_callable(np.array_equal)(union, as_callable(np.array)([1, 2, 3, 4]))

    reg_union = numpy_eager_registry.get("Union1d")
    assert as_callable(np.array_equal)(as_callable(reg_union)(np, a, b), as_callable(np.array)([1, 2, 3, 4]))

    mock_mod = types.SimpleNamespace(
        intersect1d=lambda *args, **kw: np.array([2, 3]),
        isin=lambda *args, **kw: np.array([False, True, True]),
    )

    inter = math_set._np_intersect1d_(as_mod(mock_mod), a, b)
    assert as_callable(np.array_equal)(inter, as_callable(np.array)([2, 3]))

    reg_inter = numpy_eager_registry.get("Intersect1d")
    assert as_callable(np.array_equal)(as_callable(reg_inter)(mock_mod, a, b), as_callable(np.array)([2, 3]))

    isin = math_set._np_isin_(as_mod(mock_mod), a, b)
    assert as_callable(np.array_equal)(isin, as_callable(np.array)([False, True, True]))

    reg_isin = numpy_eager_registry.get("Isin")
    assert as_callable(np.array_equal)(as_callable(reg_isin)(mock_mod, a, b), as_callable(np.array)([False, True, True]))
