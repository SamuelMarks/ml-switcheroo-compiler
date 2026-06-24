"""Assertion recording and evaluation."""

_ASSERTIONS_LIST: list[tuple[object, str]] = []


def record_assertion(condition: object, message: str = "") -> None:
    """Record an assertion for later evaluation."""
    _ASSERTIONS_LIST.append((condition, message))


def _is_iterable_non_string(c: object) -> bool:
    """Check if c is iterable but not string.

    Args:
        c: Arg.
    """
    return hasattr(c, "__iter__") and not isinstance(c, (str, bytes))  # pragma: no cover


def _evaluate_iterable(c: object) -> bool:
    """Evaluate an iterable.

    Args:
        c: Arg.
    """
    if _is_iterable_non_string(c):  # pragma: no cover
        return all(bool(x) for x in c)  # pragma: no cover
    raise ValueError(f"Could not evaluate boolean value of {type(c)}")  # pragma: no cover


def _evaluate_single_condition(cond: object) -> bool:
    """Function docstring.

    Args:
        cond: Arg.
    """
    c = cond.numpy() if hasattr(cond, "numpy") else cond

    if hasattr(c, "all") and callable(c.all):  # pragma: no branch
        return bool(c.all())

    try:  # pragma: no cover
        return bool(c)  # pragma: no cover
    except ValueError:  # pragma: no cover
        return _evaluate_iterable(c)  # pragma: no cover


def evaluate_assertions() -> None:
    """Evaluate all recorded assertions. Raises AssertionError if any fail."""
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
