"""Assertion recording and evaluation for runtime and symbolic verification."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ml_switcheroo_compiler.core.shape import TensorOrScalar

_ASSERTIONS_LIST: list[tuple[TensorOrScalar, str]] = []


def record_assertion(condition: TensorOrScalar, message: str = "") -> None:
    """Record a boolean or tensor condition assertion for deferred evaluation.

    Args:
        condition (TensorOrScalar): Boolean or tensor condition to record.
        message (str): Diagnostic error message if the assertion fails.
    """
    _ASSERTIONS_LIST.append((condition, message))


def _is_iterable_non_string(c: TensorOrScalar) -> bool:
    """Check if a value is an iterable sequence/array that is not a string or bytes.

    Args:
        c (TensorOrScalar): Value to inspect.

    Returns:
        bool: True if `c` is an `Iterable` and not `str` or `bytes`.
    """
    return isinstance(c, Iterable) and not isinstance(c, (str, bytes))


def _evaluate_iterable(c: TensorOrScalar) -> bool:
    """Evaluate truthiness across all elements of a non-string iterable.

    Args:
        c (TensorOrScalar): Iterable condition value.

    Returns:
        bool: True if all elements evaluate to True.

    Raises:
        ValueError: If `c` is not a non-string iterable.
    """
    if isinstance(c, Iterable) and not isinstance(c, (str, bytes)):
        return all(bool(x) for x in cast("Iterable[TensorOrScalar]", c))
    raise ValueError(f"Could not evaluate boolean value of {type(c)}")


def _evaluate_single_condition(cond: TensorOrScalar) -> bool:
    """Evaluate a single recorded condition (scalar, tensor, or iterable) to a boolean.

    Args:
        cond (TensorOrScalar): Recorded assertion condition.

    Returns:
        bool: True if the condition holds, else False.
    """
    c = getattr(cond, "numpy")() if hasattr(cond, "numpy") else cond

    if hasattr(c, "all") and callable(getattr(c, "all")):
        return bool(getattr(c, "all")())

    try:
        return bool(c)
    except ValueError:
        return _evaluate_iterable(c)


def evaluate_assertions() -> None:
    """Evaluate all recorded assertions and clear the assertion queue.

    Raises:
        AssertionError: If one or more recorded assertions evaluate to False.
    """
    errors = []
    for cond, msg in _ASSERTIONS_LIST:
        if not _evaluate_single_condition(cond):
            errors.append(msg)
    _ASSERTIONS_LIST.clear()
    if errors:
        raise AssertionError(chr(10).join(errors))


def clear_assertions() -> None:
    """Clear all pending recorded assertions without evaluating them."""
    _ASSERTIONS_LIST.clear()
