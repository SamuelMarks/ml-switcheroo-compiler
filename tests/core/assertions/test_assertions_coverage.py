"""Tests for test_assertions_coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

import ml_switcheroo_compiler.core.assertions as asst_mod

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_core_assertions_exhaustive() -> None:
    """Verify all condition evaluation branches, iterables, and exception handling in assertions."""
    asst_mod.clear_assertions()
    asst_mod.record_assertion(True, "Should not fail")
    asst_mod.evaluate_assertions()

    # 1. _is_iterable_non_string
    assert asst_mod._is_iterable_non_string([1, 2]) is True
    assert asst_mod._is_iterable_non_string((1, 2)) is True
    assert asst_mod._is_iterable_non_string("string") is False
    assert asst_mod._is_iterable_non_string(b"bytes") is False
    assert asst_mod._is_iterable_non_string(123) is False

    # 2. _evaluate_iterable
    assert asst_mod._evaluate_iterable([True, 1]) is True
    assert asst_mod._evaluate_iterable([True, 0]) is False
    with pytest.raises(ValueError, match="Could not evaluate"):
        asst_mod._evaluate_iterable(123)

    # 3. _evaluate_single_condition with numpy() or all()
    class WithNumpy:
        """Test class WithNumpy."""

        def numpy(self) -> list[int]:
            """Test function numpy."""
            return [1, 1]

    class WithAll:
        """Test class WithAll."""

        def all(self) -> bool:
            """Test function all."""
            return True

    class WithAllFalse:
        """Test class WithAllFalse."""

        def all(self) -> bool:
            """Test function all."""
            return False

    assert bool(cast("EagerCallableProtocol", asst_mod._evaluate_single_condition)(cast("EagerCallableProtocol", WithNumpy)())) is True
    assert bool(cast("EagerCallableProtocol", asst_mod._evaluate_single_condition)(cast("EagerCallableProtocol", WithAll)())) is True
    assert bool(cast("EagerCallableProtocol", asst_mod._evaluate_single_condition)(cast("EagerCallableProtocol", WithAllFalse)())) is False

    class ValueErrorBool:
        """Test class ValueErrorBool."""

        def __bool__(self) -> bool:
            """Test function __bool__."""
            raise ValueError("Boolean value ambiguous")

        def __iter__(self) -> EagerValue:
            """Test function __iter__."""
            return cast("EagerTensorProtocol", iter([True, True]))

    assert bool(cast("EagerCallableProtocol", asst_mod._evaluate_single_condition)(cast("EagerCallableProtocol", ValueErrorBool)())) is True

    # 4. Failure in evaluate_assertions raises AssertionError
    asst_mod.record_assertion(False, "Failed assertion 1")
    asst_mod.record_assertion(False, "Failed assertion 2")
    with pytest.raises(AssertionError, match=r"Failed assertion 1\s+Failed assertion 2"):
        asst_mod.evaluate_assertions()
