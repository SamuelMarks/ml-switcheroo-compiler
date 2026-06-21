"""Logical operations."""

from ml_switcheroo_compiler.core.dispatch import dispatch


def and_p(*args: object, **kwargs: object) -> object:
    """Execute and_p."""
    return dispatch("lax", "and_p", *args, **kwargs)


def clz(*args: object, **kwargs: object) -> object:
    """Execute clz."""
    return dispatch("lax", "clz", *args, **kwargs)


def clz_p(*args: object, **kwargs: object) -> object:
    """Execute clz_p."""
    return dispatch("lax", "clz_p", *args, **kwargs)


def eq(*args: object, **kwargs: object) -> object:
    """Execute eq."""
    return dispatch("lax", "eq", *args, **kwargs)


def eq_p(*args: object, **kwargs: object) -> object:
    """Execute eq_p."""
    return dispatch("lax", "eq_p", *args, **kwargs)


def eq_to_p(*args: object, **kwargs: object) -> object:
    """Execute eq_to_p."""
    return dispatch("lax", "eq_to_p", *args, **kwargs)


def ge(*args: object, **kwargs: object) -> object:
    """Execute ge."""
    return dispatch("lax", "ge", *args, **kwargs)


def ge_p(*args: object, **kwargs: object) -> object:
    """Execute ge_p."""
    return dispatch("lax", "ge_p", *args, **kwargs)


def gt(*args: object, **kwargs: object) -> object:
    """Execute gt."""
    return dispatch("lax", "gt", *args, **kwargs)


def gt_p(*args: object, **kwargs: object) -> object:
    """Execute gt_p."""
    return dispatch("lax", "gt_p", *args, **kwargs)


def is_finite(*args: object, **kwargs: object) -> object:
    """Execute is_finite."""
    return dispatch("lax", "is_finite", *args, **kwargs)


def is_finite_p(*args: object, **kwargs: object) -> object:
    """Execute is_finite_p."""
    return dispatch("lax", "is_finite_p", *args, **kwargs)


def le(*args: object, **kwargs: object) -> object:
    """Execute le."""
    return dispatch("lax", "le", *args, **kwargs)


def le_p(*args: object, **kwargs: object) -> object:
    """Execute le_p."""
    return dispatch("lax", "le_p", *args, **kwargs)


def le_to_p(*args: object, **kwargs: object) -> object:
    """Execute le_to_p."""
    return dispatch("lax", "le_to_p", *args, **kwargs)


def lt(*args: object, **kwargs: object) -> object:
    """Execute lt."""
    return dispatch("lax", "lt", *args, **kwargs)


def lt_p(*args: object, **kwargs: object) -> object:
    """Execute lt_p."""
    return dispatch("lax", "lt_p", *args, **kwargs)


def lt_to_p(*args: object, **kwargs: object) -> object:
    """Execute lt_to_p."""
    return dispatch("lax", "lt_to_p", *args, **kwargs)


def ne(*args: object, **kwargs: object) -> object:
    """Execute ne."""
    return dispatch("lax", "ne", *args, **kwargs)


def ne_p(*args: object, **kwargs: object) -> object:
    """Execute ne_p."""
    return dispatch("lax", "ne_p", *args, **kwargs)


def not_p(*args: object, **kwargs: object) -> object:
    """Execute not_p."""
    return dispatch("lax", "not_p", *args, **kwargs)


def or_p(*args: object, **kwargs: object) -> object:
    """Execute or_p."""
    return dispatch("lax", "or_p", *args, **kwargs)


def population_count(*args: object, **kwargs: object) -> object:
    """Execute population_count."""
    return dispatch("lax", "population_count", *args, **kwargs)


def population_count_p(*args: object, **kwargs: object) -> object:
    """Execute population_count_p."""
    return dispatch("lax", "population_count_p", *args, **kwargs)


def shift_left(*args: object, **kwargs: object) -> object:
    """Execute shift_left."""
    return dispatch("lax", "shift_left", *args, **kwargs)


def shift_left_p(*args: object, **kwargs: object) -> object:
    """Execute shift_left_p."""
    return dispatch("lax", "shift_left_p", *args, **kwargs)


def shift_right_arithmetic(*args: object, **kwargs: object) -> object:
    """Execute shift_right_arithmetic."""
    return dispatch("lax", "shift_right_arithmetic", *args, **kwargs)


def shift_right_arithmetic_p(*args: object, **kwargs: object) -> object:
    """Execute shift_right_arithmetic_p."""
    return dispatch("lax", "shift_right_arithmetic_p", *args, **kwargs)


def shift_right_logical(*args: object, **kwargs: object) -> object:
    """Execute shift_right_logical."""
    return dispatch("lax", "shift_right_logical", *args, **kwargs)


def shift_right_logical_p(*args: object, **kwargs: object) -> object:
    """Execute shift_right_logical_p."""
    return dispatch("lax", "shift_right_logical_p", *args, **kwargs)


def xor_p(*args: object, **kwargs: object) -> object:
    """Execute xor_p."""
    return dispatch("lax", "xor_p", *args, **kwargs)
