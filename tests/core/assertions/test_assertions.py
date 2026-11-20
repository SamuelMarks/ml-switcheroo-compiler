"""Unit tests for runtime and symbolic assertion recording and evaluation."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

import ml_switcheroo_compiler.core.assertions as assertions
from ml_switcheroo_compiler.core.assertions import clear_assertions, evaluate_assertions, record_assertion
from ml_switcheroo_compiler.ops.control_flow import assert_value


class _NumpyListProxy:
    """Proxy tensor-like helper returning a plain list from `.numpy()`."""

    shape: tuple[int, ...] = (2,)

    def __init__(self, values: list[int]) -> None:
        """Initialize `_NumpyListProxy`.

        Args:
            values (list[int]): Backing integer values.
        """
        self._values = values

    def numpy(self) -> list[int]:
        """Return backing list representation.

        Returns:
            list[int]: Backing list values.
        """
        return self._values


class _NumpyArrayProxy:
    """Proxy tensor-like helper returning a NumPy array from `.numpy()`."""

    shape: tuple[int, ...] = (2,)

    def __init__(self, values: list[bool]) -> None:
        """Initialize `_NumpyArrayProxy`.

        Args:
            values (list[bool]): Backing boolean values.
        """
        self._arr = np.array(values)

    def numpy(self) -> np.ndarray[tuple[int, ...], np.dtype[np.bool_]]:
        """Return backing NumPy array.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.bool_]]: Boolean NumPy array.
        """
        return self._arr


class _AllMethodCondition:
    """Tensor-like condition helper exposing `.all()`."""

    shape: tuple[int, ...] = (2,)

    def __init__(self, outcome: bool) -> None:
        """Initialize `_AllMethodCondition`.

        Args:
            outcome (bool): Result to return from `.all()`.
        """
        self._outcome = outcome

    def all(self) -> bool:
        """Return configured boolean reduction outcome.

        Returns:
            bool: Configured outcome.
        """
        return self._outcome


class _AmbiguousIterableCondition:
    """Condition helper raising `ValueError` on `bool()` and falling back to iteration."""

    shape: tuple[int, ...] = (2,)

    def __init__(self, items: tuple[bool, ...]) -> None:
        """Initialize `_AmbiguousIterableCondition`.

        Args:
            items (tuple[bool, ...]): Elements yielded during iteration.
        """
        self._items = items

    def __bool__(self) -> bool:
        """Raise `ValueError` to simulate multi-element array truth ambiguity.

        Returns:
            bool: Never returns normally.

        Raises:
            ValueError: Always raised to trigger iterable element evaluation.
        """
        raise ValueError("Boolean value of multi-element condition is ambiguous")

    def __iter__(self) -> Iterator[bool]:
        """Iterate over stored boolean elements.

        Returns:
            Iterator[bool]: Iterator over `self._items`.
        """
        return iter(self._items)


class _AmbiguousNonIterableCondition:
    """Non-iterable condition helper raising `ValueError` on `bool()`."""

    shape: tuple[int, ...] = (2,)

    def __bool__(self) -> bool:
        """Raise `ValueError` without providing an iterator fallback.

        Returns:
            bool: Never returns normally.

        Raises:
            ValueError: Always raised to trigger non-iterable error handling.
        """
        raise ValueError("Boolean value ambiguous and non-iterable")


def test_record_and_evaluate_assertions_with_arrays_and_proxies() -> None:
    """Verify `record_assertion`, `evaluate_assertions`, `.numpy()`, and `.all()` protocols."""
    clear_assertions()
    record_assertion(True, "Scalar true passes")
    record_assertion(np.array([True, True]), "All-true array passes")
    record_assertion(_NumpyListProxy([1, 1]), "Numpy list proxy passes")
    record_assertion(_NumpyArrayProxy([True, True]), "Numpy array proxy passes")
    record_assertion(_AllMethodCondition(True), "All-true protocol passes")
    evaluate_assertions()

    record_assertion(np.array([True, False]), "Array condition failed")
    with pytest.raises(AssertionError, match="Array condition failed"):
        evaluate_assertions()

    record_assertion(_NumpyArrayProxy([False, True]), "Proxy condition failed")
    with pytest.raises(AssertionError, match="Proxy condition failed"):
        evaluate_assertions()

    record_assertion(_AllMethodCondition(False), "All-false protocol failed")
    with pytest.raises(AssertionError, match="All-false protocol failed"):
        evaluate_assertions()


def test_assert_value_eager() -> None:
    """Verify `assert_value` records and raises `AssertionError` on a false condition in eager mode."""
    clear_assertions()
    with pytest.raises(AssertionError, match="Eager failed"):
        assert_value(np.array([False]), "Eager failed")
        evaluate_assertions()


def test_ambiguous_iterable_and_multiple_failure_reporting() -> None:
    """Verify iterable fallback on `ValueError` from `bool()`, non-iterable failure, and multi-error joining."""
    clear_assertions()
    record_assertion(_AmbiguousIterableCondition((True, True)), "Ambiguous all-true passes")
    evaluate_assertions()

    record_assertion(_AmbiguousIterableCondition((True, False)), "Failed assertion 1")
    record_assertion(False, "Failed assertion 2")
    with pytest.raises(AssertionError, match=r"Failed assertion 1\s+Failed assertion 2"):
        evaluate_assertions()

    clear_assertions()
    record_assertion(_AmbiguousNonIterableCondition(), "Unreachable message")
    with pytest.raises(ValueError, match="Could not evaluate boolean value"):
        evaluate_assertions()
    clear_assertions()


def test_iterable_non_string_helper() -> None:
    """Verify `_is_iterable_non_string` distinguishes sequences from strings, bytes, and scalars."""
    assert assertions._is_iterable_non_string([1, 2]) is True
    assert assertions._is_iterable_non_string((1, 2)) is True
    assert assertions._is_iterable_non_string("string") is False
    assert assertions._is_iterable_non_string(b"bytes") is False
    assert assertions._is_iterable_non_string(123) is False
