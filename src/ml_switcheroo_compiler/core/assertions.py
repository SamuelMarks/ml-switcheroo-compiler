"""Assertion recording and evaluation."""

_ASSERTIONS_LIST: list[tuple[object, str]] = []


def record_assertion(condition: object, message: str = "") -> None:
    """Record an assertion for later evaluation.

    Args:
        condition (object): The condition to record.
        message (str): The error message if it fails.
    """
    _ASSERTIONS_LIST.append((condition, message))


def _is_iterable_non_string(c: object) -> bool:
    """Check if a value is an iterable but not a string or bytes.

    Args:
        c (object): The value to check.

    Returns:
        bool: True if iterable and not string/bytes.
    """
    return hasattr(c, "__iter__") and not isinstance(c, (str, bytes))


def _evaluate_iterable(c: object) -> bool:
    """Evaluate _evaluate_iterable operation.

    Args:
        c (object): The c parameter.

    Returns:
        bool: Result.

    Raises:
        ValueError: An exception.
    """
    if _is_iterable_non_string(c):
        return all(bool(x) for x in c)
    raise ValueError(f"Could not evaluate boolean value of {type(c)}")


def _evaluate_single_condition(cond: object) -> bool:
    """Evaluate _evaluate_single_condition operation.

    Args:
        cond (object): The cond parameter.

    Returns:
        bool: Result.
    """
    c = cond.numpy() if hasattr(cond, "numpy") else cond

    if hasattr(c, "all") and callable(c.all):
        return bool(c.all())

    try:
        return bool(c)
    except ValueError:
        return _evaluate_iterable(c)


def evaluate_assertions() -> None:
    """Evaluate evaluate_assertions operation.

    Raises:
        AssertionError: An exception.
    """
    errors = []
    for cond, msg in _ASSERTIONS_LIST:
        if not _evaluate_single_condition(cond):
            errors.append(msg)
    _ASSERTIONS_LIST.clear()
    if errors:
        raise AssertionError("\n".join(errors))


def clear_assertions() -> None:
    """Clear all recorded assertions."""
    _ASSERTIONS_LIST.clear()
