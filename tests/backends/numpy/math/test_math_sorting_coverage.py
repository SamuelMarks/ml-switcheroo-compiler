"""Tests for test_math_sorting_coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_sorting as math_sorting

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


def test_math_sorting() -> None:
    """Test ArgPartition, Lexsort, and Median directly and via registry."""
    arr = np.array([3, 1, 2, 5, 4])
    part = math_sorting._np_argpartition(np, arr, 2)
    assert (
        cast("EagerCallableProtocol", getattr(part, "__getitem__"))(2) == 2
        or cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", getattr(arr, "__getitem__"))(cast("EagerTensorProtocol", part)[:2]), [1, 2])
        or cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", getattr(arr, "__getitem__"))(cast("EagerTensorProtocol", part)[:2]), [2, 1])
    )

    reg_part = numpy_eager_registry.get("ArgPartition")
    part_reg = cast("EagerCallableProtocol", reg_part)(np, arr, 2)
    assert cast("EagerCallableProtocol", np.array_equal)(part, part_reg)

    first = np.array([1, 2, 1, 2])
    second = np.array([4, 1, 3, 2])
    lex = math_sorting._np_lexsort_(np, (first, second))
    reg_lex = numpy_eager_registry.get("Lexsort")
    assert cast("EagerCallableProtocol", np.array_equal)(lex, cast("EagerCallableProtocol", reg_lex)(np, (first, second)))

    med_arr = np.array([1.0, 5.0, 3.0])
    assert math_sorting._np_median_(np, med_arr) == 3.0
    reg_med = numpy_eager_registry.get("Median")
    assert cast("EagerCallableProtocol", reg_med)(np, med_arr) == 3.0
