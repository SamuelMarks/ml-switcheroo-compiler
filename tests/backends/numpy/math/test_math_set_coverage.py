"""Tests for test_math_set_coverage."""

from __future__ import annotations

import types
import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_set as math_set
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mod

if typing.TYPE_CHECKING:
    pass


def test_math_set() -> None:
    """Test Union1d, Intersect1d, and Isin directly and via registry."""
    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array([2, 3, 4], dtype=np.int64)

    union = math_set._np_union1d(ensure_eager_tensor(np), a, b)
    assert ensure_callable(np.array_equal)(union, ensure_callable(np.array)([1, 2, 3, 4]))

    reg_union = numpy_eager_registry.get("Union1d")
    assert ensure_callable(np.array_equal)(ensure_callable(reg_union)(np, a, b), ensure_callable(np.array)([1, 2, 3, 4]))

    mock_mod = types.SimpleNamespace(
        intersect1d=lambda *args, **kw: np.array([2, 3]),
        isin=lambda *args, **kw: np.array([False, True, True]),
    )

    inter = math_set._np_intersect1d_(ensure_mod(mock_mod), a, b)
    assert ensure_callable(np.array_equal)(inter, ensure_callable(np.array)([2, 3]))

    reg_inter = numpy_eager_registry.get("Intersect1d")
    assert ensure_callable(np.array_equal)(ensure_callable(reg_inter)(mock_mod, a, b), ensure_callable(np.array)([2, 3]))

    isin = math_set._np_isin_(ensure_mod(mock_mod), a, b)
    assert ensure_callable(np.array_equal)(isin, ensure_callable(np.array)([False, True, True]))

    reg_isin = numpy_eager_registry.get("Isin")
    assert ensure_callable(np.array_equal)(ensure_callable(reg_isin)(mock_mod, a, b), ensure_callable(np.array)([False, True, True]))
