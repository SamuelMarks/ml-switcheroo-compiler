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
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol

if typing.TYPE_CHECKING:
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
    assert _intersect1d(typing.cast("BackendModuleProtocol", sb), [1, 2], [2, 3]) == "intersect1d_val"
    assert _np_union1d(typing.cast("BackendModuleProtocol", sb), [1], [2]) == "union1d_val"
    assert _np_unique(typing.cast("BackendModuleProtocol", sb), [1, 1, 2]) == "unique_val"
    assert _np_uniqueall(typing.cast("BackendModuleProtocol", sb), [1, 1, 2]) == "uniqueall_val"
    assert _np_uniquecounts(typing.cast("BackendModuleProtocol", sb), [1, 1, 2]) == "uniquecounts_val"
    assert _np_uniquevalues(typing.cast("BackendModuleProtocol", sb), [1, 1, 2]) == "uniquevalues_val"

    # Fallback to NumPy when backend lacks the methods
    class EmptyBackend:
        """Empty backend."""

        pass

    eb = EmptyBackend()
    assert typing.cast("EagerCallableProtocol", list)(typing.cast("EagerCallableProtocol", _np_union1d)(eb, [1, 2], [2, 3])) == [1, 2, 3]
    assert typing.cast("EagerCallableProtocol", list)(typing.cast("EagerCallableProtocol", _np_unique)(eb, [3, 1, 2, 1])) == [1, 2, 3]

    vals, idx, inv, counts = typing.cast("EagerCallableProtocol", _np_uniqueall)(eb, typing.cast("EagerCallableProtocol", np.array)([2, 1, 2]))
    assert typing.cast("EagerCallableProtocol", list)(vals) == [1, 2]
    assert typing.cast("EagerCallableProtocol", len)(idx) == 2
    assert typing.cast("EagerCallableProtocol", len)(inv) == 3
    assert typing.cast("EagerCallableProtocol", list)(counts) == [1, 2]

    u_vals, u_counts = typing.cast("EagerCallableProtocol", _np_uniquecounts)(eb, typing.cast("EagerCallableProtocol", np.array)([5, 5, 6]))
    assert typing.cast("EagerCallableProtocol", list)(u_vals) == [5, 6]
    assert typing.cast("EagerCallableProtocol", list)(u_counts) == [2, 1]

    val_only = _np_uniquevalues(typing.cast("BackendModuleProtocol", eb), np.array([7, 8, 7]))
    assert typing.cast("EagerCallableProtocol", list)(val_only) == [7, 8]

    # Registry verification
    for name in ["Intersect1d", "Union1d", "Unique", "UniqueAll", "UniqueCounts", "UniqueValues"]:
        assert global_eager_registry.get(name) is not None
