"""Tests for test_core_math_set."""

from __future__ import annotations

import typing

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
from tests.testing_utils import as_callable, as_mod

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_math_set_coverage() -> None:
    """Test all branches and lines of math_set.

    Returns:
        None
    """

    class SetBackend:
        """Backend with set operations."""

        def intersect1d(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock intersect1d.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "intersect1d_val"

        def union1d(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock union1d.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "union1d_val"

        def unique(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock unique.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "unique_val"

        def uniqueall(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock uniqueall.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "uniqueall_val"

        def uniquecounts(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock uniquecounts.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "uniquecounts_val"

        def uniquevalues(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock uniquevalues.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "uniquevalues_val"

    sb = SetBackend()
    assert _intersect1d(as_mod(sb), [1, 2], [2, 3]) == "intersect1d_val"
    assert _np_union1d(as_mod(sb), [1], [2]) == "union1d_val"
    assert _np_unique(as_mod(sb), [1, 1, 2]) == "unique_val"
    assert _np_uniqueall(as_mod(sb), [1, 1, 2]) == "uniqueall_val"
    assert _np_uniquecounts(as_mod(sb), [1, 1, 2]) == "uniquecounts_val"
    assert _np_uniquevalues(as_mod(sb), [1, 1, 2]) == "uniquevalues_val"

    # Fallback to NumPy when backend lacks the methods
    class EmptyBackend:
        """Empty backend."""

        pass

    eb = EmptyBackend()
    assert as_callable(list)(as_callable(_np_union1d)(eb, [1, 2], [2, 3])) == [1, 2, 3]
    assert as_callable(list)(as_callable(_np_unique)(eb, [3, 1, 2, 1])) == [1, 2, 3]

    vals, idx, inv, counts = as_callable(_np_uniqueall)(eb, as_callable(np.array)([2, 1, 2]))
    assert as_callable(list)(vals) == [1, 2]
    assert as_callable(len)(idx) == 2
    assert as_callable(len)(inv) == 3
    assert as_callable(list)(counts) == [1, 2]

    u_vals, u_counts = as_callable(_np_uniquecounts)(eb, as_callable(np.array)([5, 5, 6]))
    assert as_callable(list)(u_vals) == [5, 6]
    assert as_callable(list)(u_counts) == [2, 1]

    val_only = _np_uniquevalues(as_mod(eb), np.array([7, 8, 7]))
    assert as_callable(list)(val_only) == [7, 8]

    # Registry verification
    for name in ["Intersect1d", "Union1d", "Unique", "UniqueAll", "UniqueCounts", "UniqueValues"]:
        assert global_eager_registry.get(name) is not None
