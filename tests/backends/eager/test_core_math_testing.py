"""Tests for test_core_math_testing."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

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

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def _test_math_testing_coverage_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_math_testing_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class AllcloseBackend:
        """Backend with allclose."""

        def allclose(self, a: EagerValue, b: EagerValue, **kwargs: EagerValue) -> bool:
            """Mock allclose.

            Args:
                a: Test parameter `a`.
                b: Test parameter `b`.
                kwargs: Test parameter `kwargs`.
            """
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
            """Initialize with value.

            Args:
                val: Test parameter `val`.
            """
            self.val = val

        def item(self) -> float:
            """Return item val."""
            return self.val

    getattr(_state, "update")({"AllcloseBackend": AllcloseBackend, "EmptyBackend": EmptyBackend, "ItemObj": ItemObj, "acb": acb})
    getattr(_state, "update")({"eb": eb})
    _state.update(locals())


def _test_math_testing_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_math_testing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    EmptyBackend, ItemObj, eb = _state["EmptyBackend"], _state["ItemObj"], _state["eb"]

    class ToListObj:
        """Object with tolist() method."""

        def __init__(self, val: list[float]) -> None:
            """Initialize with list.

            Args:
                val: Test parameter `val`.
            """
            self.val = val

        def tolist(self) -> float:
            """Return first item from list."""
            return self.val[0]

    class DataObj:
        """Object with data attribute."""

        def __init__(self, data: EagerValue) -> None:
            """Initialize with data.

            Args:
                data: Test parameter `data`.
            """
            self.data = data

    assert (
        bool(
            cast("EagerCallableProtocol", _allclose)(
                eb,
                [1.0],
                [1.0],
                rtol=ItemObj(1e-4),
                atol=cast("EagerCallableProtocol", DataObj)(cast("EagerCallableProtocol", ToListObj)([1e-6])),
                equal_nan=DataObj(False),
            )
        )
        is True
    )

    # 2. _array_equiv
    class ArrayEquivBackend:
        """Backend with allclose."""

        def allclose(self, a1: EagerValue, a2: EagerValue) -> str:
            """Mock allclose.

            Args:
                a1: Test parameter `a1`.
                a2: Test parameter `a2`.
            """
            return "allclose_equiv"

    assert _array_equiv(ArrayEquivBackend(), [1], [1]) == "allclose_equiv"
    assert _array_equiv(EmptyBackend(), [1], [1]) is True

    # 3. _assert
    # Case A: condition is True -> returns None
    assert _assert(eb, True) is None
    getattr(_state, "update")({"ArrayEquivBackend": ArrayEquivBackend, "DataObj": DataObj, "ToListObj": ToListObj})
    _state.update(locals())


def _test_math_testing_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_math_testing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    eb = _state["eb"]
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

        def promote_types(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock promote_types.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "promoted"

        def result_type(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock result_type.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "result_type"

    tb = TypeBackend()
    assert _promotetypes(tb, "float32", "float64") == "promoted"
    assert _resulttype(tb, "int32", "float32") == "result_type"
    getattr(_state, "update")({"TypeBackend": TypeBackend, "exc_info": exc_info, "exc_info_data": exc_info_data, "exc_info_empty_data": exc_info_empty_data})
    getattr(_state, "update")({"tb": tb})
    _state.update(locals())


def _test_math_testing_coverage_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_math_testing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    for name in ["Allclose", "ArrayEquiv", "Assert", "PromoteTypes", "ResultType"]:
        assert global_eager_registry.get(name) is not None


def test_math_testing_coverage() -> None:
    """Test all branches and lines of math_testing.

    Returns:
        None
    """
    _state: dict[str, EagerTensorProtocol] = {}
    _test_math_testing_coverage_part_1(_state)
    _test_math_testing_coverage_part_2(_state)
    _test_math_testing_coverage_part_3(_state)
    _test_math_testing_coverage_part_4(_state)
    assert len(_state) > 0
