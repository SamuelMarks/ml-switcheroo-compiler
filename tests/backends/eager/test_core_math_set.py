"""Tests for test_core_math_set."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

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

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerValue


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
    assert cast("EagerCallableProtocol", list)(cast("EagerCallableProtocol", _np_union1d)(eb, [1, 2], [2, 3])) == [1, 2, 3]
    assert cast("EagerCallableProtocol", list)(cast("EagerCallableProtocol", _np_unique)(eb, [3, 1, 2, 1])) == [1, 2, 3]

    vals, idx, inv, counts = cast("EagerCallableProtocol", _np_uniqueall)(eb, cast("EagerCallableProtocol", np.array)([2, 1, 2]))
    assert cast("EagerCallableProtocol", list)(vals) == [1, 2]
    assert cast("EagerCallableProtocol", len)(idx) == 2
    assert cast("EagerCallableProtocol", len)(inv) == 3
    assert cast("EagerCallableProtocol", list)(counts) == [1, 2]

    u_vals, u_counts = cast("EagerCallableProtocol", _np_uniquecounts)(eb, cast("EagerCallableProtocol", np.array)([5, 5, 6]))
    assert cast("EagerCallableProtocol", list)(u_vals) == [5, 6]
    assert cast("EagerCallableProtocol", list)(u_counts) == [2, 1]

    val_only = _np_uniquevalues(eb, np.array([7, 8, 7]))
    assert cast("EagerCallableProtocol", list)(val_only) == [7, 8]

    # Registry verification
    for name in ["Intersect1d", "Union1d", "Unique", "UniqueAll", "UniqueCounts", "UniqueValues"]:
        assert global_eager_registry.get(name) is not None
