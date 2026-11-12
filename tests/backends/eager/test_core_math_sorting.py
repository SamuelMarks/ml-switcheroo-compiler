"""Tests for test_core_math_sorting."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_sort import (
    _argsort,
    _lexsort,
    _searchsorted,
    _sortcomplex,
)
from ml_switcheroo_compiler.backends.eager.core_math_ops.math_sorting import (
    _argpartition,
    _median,
    _np_partition,
    _percentile,
    _quantile,
)
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from tests.testing_utils import ensure_eager_tensor, ensure_mod

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_math_sort_coverage() -> None:
    """Test 100% lines and branches of math_sort.

    Returns:
        None
    """

    class SortBackend:
        """Mock backend with sort operations."""

        def argsort(self, a: EagerValue, axis: int = -1) -> str:
            """Mock argsort.

            Args:
                a: Test parameter `a`.
                axis: Test parameter `axis`.

            Returns:
                str: Computed output value from `argsort`.
            """
            return f"argsort_{axis}"

        def lexsort(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock lexsort.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `lexsort`.
            """
            return "lexsort_val"

        def searchsorted(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock searchsorted.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `searchsorted`.
            """
            return "searchsorted_val"

        def sort_complex(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock sort_complex.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `sort_complex`.
            """
            return "sort_complex_val"

    sb = SortBackend()
    assert _argsort(ensure_mod(sb), [3, 1, 2], axis=1) == "argsort_1"
    assert _lexsort(ensure_mod(sb), ([1, 2], [3, 4])) == "lexsort_val"
    assert _searchsorted(ensure_mod(sb), [1, 2, 3], 2) == "searchsorted_val"
    assert _sortcomplex(ensure_mod(sb), [1 + 2j, 1 - 2j]) == "sort_complex_val"

    # Registry
    assert global_eager_registry.get("Argsort") is not None
    assert global_eager_registry.get("Lexsort") is not None
    assert global_eager_registry.get("Searchsorted") is not None
    assert global_eager_registry.get("SortComplex") is not None


def test_math_sorting_coverage() -> None:
    """Test 100% lines and branches of math_sorting.

    Returns:
        None
    """

    # _argpartition
    # Case A: backend has argsort
    class ArgpartBackend:
        """Backend with argsort."""

        def argsort(self, a: EagerValue, axis: int = -1) -> str:
            """Mock argsort.

            Args:
                a: Test parameter `a`.
                axis: Test parameter `axis`.

            Returns:
                str: Computed output value from `argsort`.
            """
            return f"argpartition_{axis}"

    assert _argpartition(ensure_mod(ArgpartBackend()), [3, 1, 2], kth=1, axis=0) == "argpartition_0"

    # Case B: backend lacks argsort -> returns a directly
    class EmptyBackend:
        """Empty backend."""

        pass

    assert _argpartition(ensure_mod(EmptyBackend()), "original_a", kth=1) == "original_a"

    # _median, _percentile, _quantile
    class StatBackend:
        """Backend with median, percentile, quantile."""

        def median(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock median.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `median`.
            """
            return "median_val"

        def percentile(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock percentile.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `percentile`.
            """
            return "percentile_val"

        def quantile(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock quantile.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `quantile`.
            """
            return "quantile_val"

    st = StatBackend()
    assert _median(ensure_mod(st), [1, 2, 3]) == "median_val"
    assert _percentile(ensure_mod(st), [1, 2, 3], 50) == "percentile_val"
    assert _quantile(ensure_mod(st), [1, 2, 3], 0.5) == "quantile_val"

    # _np_partition
    # Case A: backend has partition
    class PartitionBackend:
        """Backend with partition."""

        def partition(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock partition.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `partition`.
            """
            return "partition_val"

    assert _np_partition(ensure_mod(PartitionBackend()), [3, 1, 2], 1) == "partition_val"

    # Case B: backend lacks partition -> fallback to np.partition
    part_res = _np_partition(ensure_mod(EmptyBackend()), np.array([3, 1, 2]), 1)
    assert ensure_eager_tensor(part_res)[1] == 2

    # Registry
    assert global_eager_registry.get("Argpartition") is not None
    assert global_eager_registry.get("Median") is not None
    assert global_eager_registry.get("Percentile") is not None
    assert global_eager_registry.get("Quantile") is not None
    assert global_eager_registry.get("Partition") is not None
