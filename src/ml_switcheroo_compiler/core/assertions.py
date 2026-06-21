"""Assertion recording and evaluation."""

_ASSERTIONS_LIST: list[tuple[object, str]] = []


def record_assertion(condition: object, message: str = "") -> None:
    """Record an assertion for later evaluation."""
    _ASSERTIONS_LIST.append((condition, message))


def _evaluate_single_condition(cond: object) -> bool:
    if hasattr(cond, "numpy"):
        c = cond.numpy()
    else:
        c = cond

    if hasattr(c, "all") and callable(c.all):
        return bool(c.all())

    try:
        return bool(c)
    except ValueError:
        if hasattr(c, "__iter__") and not isinstance(c, (str, bytes)):
            return all(bool(x) for x in c)
        raise


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
