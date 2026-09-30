"""Tests for test_math_logical_reductions_coverage."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import math_logical_reductions as logical_reductions


def test_math_logical_reductions() -> None:
    """Test All and CountNonzero reductions directly and via registry."""
    arr_all_true = np.array([True, True, True])
    arr_some_false = np.array([True, False, True])

    assert logical_reductions._np_all(np, arr_all_true) == True
    assert logical_reductions._np_all(np, arr_some_false) == False

    reg_all = numpy_eager_registry.get("All")
    assert reg_all(np, arr_all_true) == True
    assert reg_all(np, arr_some_false) == False

    arr_counts = np.array([[0, 1, 2], [3, 0, 0]])
    assert logical_reductions._np_count_nonzero(np, arr_counts) == 3

    reg_count = numpy_eager_registry.get("CountNonzero")
    assert reg_count(np, arr_counts) == 3
