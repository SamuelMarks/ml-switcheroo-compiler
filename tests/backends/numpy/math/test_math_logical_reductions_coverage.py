"""Tests for test_math_logical_reductions_coverage."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import math_logical_reductions as logical_reductions
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    pass


def test_math_logical_reductions() -> None:
    """Test All and CountNonzero reductions directly and via registry."""
    arr_all_true = np.array([True, True, True])
    arr_some_false = np.array([True, False, True])

    assert logical_reductions._np_all(ensure_eager_tensor(np), arr_all_true)
    assert not logical_reductions._np_all(ensure_eager_tensor(np), arr_some_false)

    reg_all = numpy_eager_registry.get("All")
    assert ensure_callable(reg_all)(np, arr_all_true)
    assert not ensure_callable(reg_all)(np, arr_some_false)

    arr_counts = np.array([[0, 1, 2], [3, 0, 0]])
    assert logical_reductions._np_count_nonzero(ensure_eager_tensor(np), arr_counts) == 3

    reg_count = numpy_eager_registry.get("CountNonzero")
    assert ensure_callable(reg_count)(np, arr_counts) == 3
