"""Unit tests for runtime and symbolic assertion recording and evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import ml_switcheroo_compiler.core.assertions as assertions
from ml_switcheroo_compiler.core.assertions import clear_assertions, evaluate_assertions, record_assertion
from ml_switcheroo_compiler.ops.control_flow import assert_value

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerValue


def test_assertions() -> None:
    """Verify `record_assertion`, `evaluate_assertions`, and proxy `.numpy()` unwrapping."""
    clear_assertions()
    record_assertion(np.array([True, True]), "Should pass")
    evaluate_assertions()
    record_assertion(np.array([True, False]), "Should fail")
    with pytest.raises(AssertionError, match="Should fail"):
        evaluate_assertions()

    class DummyProxy:
        """Proxy test helper exposing `.numpy()`."""

        def __init__(self, val: EagerValue) -> None:
            """Initialize DummyProxy.

            Args:
                val (EagerValue): Wrapped array value.
            """
            self.val = val

        def numpy(self) -> EagerValue:
            """Return the wrapped array value.

            Returns:
                EagerValue: Wrapped array.
            """
            return self.val

    cast("EagerCallableProtocol", record_assertion)(
        cast("EagerCallableProtocol", DummyProxy)(cast("EagerCallableProtocol", np.array)([False])),
        "Proxy failed",
    )
    with pytest.raises(AssertionError, match="Proxy failed"):
        evaluate_assertions()


def test_assert_value_eager() -> None:
    """Verify `assert_value` raises `AssertionError` on a false condition in eager mode."""
    clear_assertions()
    with pytest.raises(AssertionError, match="Eager failed"):
        assert_value(np.array([False]), "Eager failed")
        evaluate_assertions()


def test_assertions_coverage() -> None:
    """Verify `_is_iterable_non_string`, `_evaluate_iterable`, and `_evaluate_single_condition`."""
    assertions.clear_assertions()

    assertions.record_assertion(True, "Should not fail")
    assertions.record_assertion(False, "Should fail")

    assert assertions._is_iterable_non_string([1, 2]) is True
    assert assertions._is_iterable_non_string("string") is False
    assert assertions._is_iterable_non_string(b"bytes") is False
    assert assertions._is_iterable_non_string(1) is False

    assert assertions._evaluate_iterable([True, True]) is True
    assert assertions._evaluate_iterable([True, False]) is False
    with pytest.raises(ValueError, match="Could not evaluate boolean value"):
        assertions._evaluate_iterable(1)

    class NumpyCond:
        """Condition helper returning an all-True NumPy array."""

        def numpy(self) -> EagerValue:
            """Return an all-True NumPy array.

            Returns:
                EagerValue: Boolean array `[True, True]`.
            """
            return np.array([True, True])

    assert bool(cast("EagerCallableProtocol", assertions._evaluate_single_condition)(cast("EagerCallableProtocol", NumpyCond)())) is True

    class NumpyCondFalse:
        """Condition helper returning a mixed NumPy array."""

        def numpy(self) -> EagerValue:
            """Return a mixed boolean NumPy array.

            Returns:
                EagerValue: Boolean array `[True, False]`.
            """
            return np.array([True, False])

    assert bool(cast("EagerCallableProtocol", assertions._evaluate_single_condition)(cast("EagerCallableProtocol", NumpyCondFalse)())) is False

    assert assertions._evaluate_single_condition(True) is True
    assert assertions._evaluate_single_condition(False) is False

    class ValueErrCond:
        """Condition helper raising `ValueError` on `bool()` and falling back to `__iter__`."""

        def __bool__(self) -> bool:
            """Raise `ValueError` to mimic multi-element array truth ambiguity.

            Raises:
                ValueError: Always raised.
            """
            raise ValueError("ambiguous truth value")

        def __iter__(self) -> Iterator[EagerValue]:
            """Yield boolean elements.

            Yields:
                EagerValue: True elements.
            """
            yield True
            yield True

    assert bool(cast("EagerCallableProtocol", assertions._evaluate_single_condition)(cast("EagerCallableProtocol", ValueErrCond)())) is True

    assertions.clear_assertions()
    assertions.record_assertion(True, "All good")
    assertions.evaluate_assertions()

    assertions.record_assertion(False, "Oh no")
    with pytest.raises(AssertionError, match="Oh no"):
        assertions.evaluate_assertions()
