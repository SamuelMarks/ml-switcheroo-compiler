"""Tests for test_core_math_testing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_testing import (
    _allclose,
    _array_equiv,
    _assert,
    _promotetypes,
    _resulttype,
)
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


def test_math_testing_coverage() -> None:
    """Test all branches and lines of math_testing.

    Returns:
        None
    """

    # 1. _allclose
    # Case A: backend has allclose
    class AllcloseBackend:
        """Backend with allclose."""

        def allclose(self, a: Any, b: Any, **kwargs: Any) -> bool:
            """Mock allclose."""
            return True

    acb = AllcloseBackend()
    assert _allclose(acb, [1.0], [1.0]) is True

    # Case B: backend lacks allclose -> fallback to np.allclose
    class EmptyBackend:
        """Empty backend."""

        pass

    eb = EmptyBackend()

    class ItemObj:
        """Object with item() method."""

        def __init__(self, val: float) -> None:
            """Initialize with value."""
            self.val = val

        def item(self) -> float:
            """Return item val."""
            return self.val

    class ToListObj:
        """Object with tolist() method."""

        def __init__(self, val: list[float]) -> None:
            """Initialize with list."""
            self.val = val

        def tolist(self) -> float:
            """Return first item from list."""
            return self.val[0]

    class DataObj:
        """Object with data attribute."""

        def __init__(self, data: Any) -> None:
            """Initialize with data."""
            self.data = data

    assert (
        _allclose(
            eb,
            [1.0],
            [1.0],
            rtol=ItemObj(1e-4),
            atol=DataObj(ToListObj([1e-6])),
            equal_nan=DataObj(False),
        )
        is True
    )

    # 2. _array_equiv
    class ArrayEquivBackend:
        """Backend with allclose."""

        def allclose(self, a1: Any, a2: Any) -> str:
            """Mock allclose."""
            return "allclose_equiv"

    assert _array_equiv(ArrayEquivBackend(), [1], [1]) == "allclose_equiv"
    assert _array_equiv(EmptyBackend(), [1], [1]) is True

    # 3. _assert
    # Case A: condition is True -> returns None
    assert _assert(eb, True) is None
    assert _assert(eb, [True, True]) is None

    # Case B: condition evaluates to False without data
    with pytest.raises(AssertionError) as exc_info:
        _assert(eb, False, node_id="TestNode", message="Value check failed")
    assert "Assertion failed in node 'TestNode': Value check failed" in str(exc_info.value)

    # Case C: condition evaluates to False with data
    with pytest.raises(AssertionError) as exc_info_data:
        _assert(
            eb,
            np.array([False]),
            data=np.array([10, 20, 30, 40]),
            summarize=2,
            node_id="DataNode",
            message="Data fail",
        )
    assert "Data sample: [10, 20]" in str(exc_info_data.value)

    # Case D: condition evaluates to False with empty flat data
    with pytest.raises(AssertionError) as exc_info_empty_data:
        _assert(eb, False, data=np.array([]), node_id="EmptyDataNode")
    assert "Data sample: []" in str(exc_info_empty_data.value)

    # 4. _promotetypes and _resulttype
    class TypeBackend:
        """Backend with type inference methods."""

        def promote_types(self, *args: Any, **kwargs: Any) -> str:
            """Mock promote_types."""
            return "promoted"

        def result_type(self, *args: Any, **kwargs: Any) -> str:
            """Mock result_type."""
            return "result_type"

    tb = TypeBackend()
    assert _promotetypes(tb, "float32", "float64") == "promoted"
    assert _resulttype(tb, "int32", "float32") == "result_type"

    # Registry verification
    for name in ["Allclose", "ArrayEquiv", "Assert", "PromoteTypes", "ResultType"]:
        assert global_eager_registry.get(name) is not None
