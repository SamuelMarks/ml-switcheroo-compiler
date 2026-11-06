"""Tests for test_math_sorting_coverage."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_sorting as math_sorting
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    pass


def test_math_sorting() -> None:
    """Test ArgPartition, Lexsort, and Median directly and via registry."""
    arr = np.array([3, 1, 2, 5, 4])
    part = math_sorting._np_argpartition(ensure_eager_tensor(np), arr, 2)
    assert ensure_eager_tensor(part)[2] == 2 or ensure_callable(np.array_equal)(ensure_eager_tensor(arr)[ensure_eager_tensor(part)[:2]], [1, 2]) or ensure_callable(np.array_equal)(ensure_eager_tensor(arr)[ensure_eager_tensor(part)[:2]], [2, 1])

    reg_part = numpy_eager_registry.get("ArgPartition")
    part_reg = ensure_callable(reg_part)(np, arr, 2)
    assert ensure_callable(np.array_equal)(part, part_reg)

    first = np.array([1, 2, 1, 2])
    second = np.array([4, 1, 3, 2])
    lex = math_sorting._np_lexsort_(ensure_eager_tensor(np), (first, second))
    reg_lex = numpy_eager_registry.get("Lexsort")
    assert ensure_callable(np.array_equal)(lex, ensure_callable(reg_lex)(np, (first, second)))

    med_arr = np.array([1.0, 5.0, 3.0])
    assert math_sorting._np_median_(ensure_eager_tensor(np), med_arr) == 3.0
    reg_med = numpy_eager_registry.get("Median")
    assert ensure_callable(reg_med)(np, med_arr) == 3.0
