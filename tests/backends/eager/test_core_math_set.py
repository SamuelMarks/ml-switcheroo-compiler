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
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry, unpack_quad
from tests.testing_utils import MockTestStub, coerce_operand_list, unpack_pair

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_math_set_coverage() -> None:
    """Test all branches and lines of math_set.

    Returns:
        None
    """

    class SetBackend(MockTestStub):
        """Backend with set operations."""

        def intersect1d(self, *args: BackendArg, **kwargs: BackendArg) -> str:
            """Mock intersect1d.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `intersect1d`.
            """
            return "intersect1d_val"

        def union1d(self, *args: BackendArg, **kwargs: BackendArg) -> str:
            """Mock union1d.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `union1d`.
            """
            return "union1d_val"

        def unique(self, *args: BackendArg, **kwargs: BackendArg) -> str:
            """Mock unique.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `unique`.
            """
            return "unique_val"

        def uniqueall(self, *args: BackendArg, **kwargs: BackendArg) -> str:
            """Mock uniqueall.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `uniqueall`.
            """
            return "uniqueall_val"

        def uniquecounts(self, *args: BackendArg, **kwargs: BackendArg) -> str:
            """Mock uniquecounts.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `uniquecounts`.
            """
            return "uniquecounts_val"

        def uniquevalues(self, *args: BackendArg, **kwargs: BackendArg) -> str:
            """Mock uniquevalues.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `uniquevalues`.
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
    class EmptyBackend(MockTestStub):
        """Empty backend."""

        pass

    eb = EmptyBackend()
    assert coerce_operand_list(_np_union1d(eb, [1, 2], [2, 3])) == [1, 2, 3]
    assert coerce_operand_list(_np_unique(eb, [3, 1, 2, 1])) == [1, 2, 3]

    vals, idx, inv, counts = unpack_quad(_np_uniqueall(eb, np.array([2, 1, 2])))
    assert coerce_operand_list(vals) == [1, 2]
    assert len(coerce_operand_list(idx)) == 2
    assert len(coerce_operand_list(inv)) == 3
    assert coerce_operand_list(counts) == [1, 2]

    u_vals, u_counts = unpack_pair(_np_uniquecounts(eb, np.array([5, 5, 6])))
    assert coerce_operand_list(u_vals) == [5, 6]
    assert coerce_operand_list(u_counts) == [2, 1]

    val_only = _np_uniquevalues(eb, np.array([7, 8, 7]))
    assert coerce_operand_list(val_only) == [7, 8]

    # Registry verification
    for name in ["Intersect1d", "Union1d", "Unique", "UniqueAll", "UniqueCounts", "UniqueValues"]:
        assert global_eager_registry.get(name) is not None
