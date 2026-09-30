"""Tests for test_core_math_set."""

from __future__ import annotations

from typing import Any

import numpy as np

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_set import (
    _intersect1d,
    _np_union1d,
    _np_unique,
    _np_uniqueall,
    _np_uniquecounts,
    _np_uniquevalues,
)
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


def test_math_set_coverage() -> None:
    """Test all branches and lines of math_set.

    Returns:
        None
    """

    class SetBackend:
        """Backend with set operations."""

        def intersect1d(self, *args: Any, **kwargs: Any) -> str:
            """Mock intersect1d."""
            return "intersect1d_val"

        def union1d(self, *args: Any, **kwargs: Any) -> str:
            """Mock union1d."""
            return "union1d_val"

        def unique(self, *args: Any, **kwargs: Any) -> str:
            """Mock unique."""
            return "unique_val"

        def uniqueall(self, *args: Any, **kwargs: Any) -> str:
            """Mock uniqueall."""
            return "uniqueall_val"

        def uniquecounts(self, *args: Any, **kwargs: Any) -> str:
            """Mock uniquecounts."""
            return "uniquecounts_val"

        def uniquevalues(self, *args: Any, **kwargs: Any) -> str:
            """Mock uniquevalues."""
            return "uniquevalues_val"

    sb = SetBackend()
    assert _intersect1d(sb, [1, 2], [2, 3]) == "intersect1d_val"
    assert _np_union1d(sb, [1], [2]) == "union1d_val"
    assert _np_unique(sb, [1, 1, 2]) == "unique_val"
    assert _np_uniqueall(sb, [1, 1, 2]) == "uniqueall_val"
    assert _np_uniquecounts(sb, [1, 1, 2]) == "uniquecounts_val"
    assert _np_uniquevalues(sb, [1, 1, 2]) == "uniquevalues_val"

    # Fallback to NumPy when backend lacks the methods
    class EmptyBackend:
        """Empty backend."""

        pass

    eb = EmptyBackend()
    assert list(_np_union1d(eb, [1, 2], [2, 3])) == [1, 2, 3]
    assert list(_np_unique(eb, [3, 1, 2, 1])) == [1, 2, 3]

    vals, idx, inv, counts = _np_uniqueall(eb, np.array([2, 1, 2]))
    assert list(vals) == [1, 2]
    assert len(idx) == 2
    assert len(inv) == 3
    assert list(counts) == [1, 2]

    u_vals, u_counts = _np_uniquecounts(eb, np.array([5, 5, 6]))
    assert list(u_vals) == [5, 6]
    assert list(u_counts) == [2, 1]

    val_only = _np_uniquevalues(eb, np.array([7, 8, 7]))
    assert list(val_only) == [7, 8]

    # Registry verification
    for name in ["Intersect1d", "Union1d", "Unique", "UniqueAll", "UniqueCounts", "UniqueValues"]:
        assert global_eager_registry.get(name) is not None
