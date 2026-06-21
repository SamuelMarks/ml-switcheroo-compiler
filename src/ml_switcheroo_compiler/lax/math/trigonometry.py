"""Trigonometry operations."""

from ml_switcheroo_compiler.core.dispatch import dispatch


def acos_p(*args: object, **kwargs: object) -> object:
    """Execute acos_p."""
    return dispatch("lax", "acos_p", *args, **kwargs)


def acosh_p(*args: object, **kwargs: object) -> object:
    """Execute acosh_p."""
    return dispatch("lax", "acosh_p", *args, **kwargs)


def asin_p(*args: object, **kwargs: object) -> object:
    """Execute asin_p."""
    return dispatch("lax", "asin_p", *args, **kwargs)


def asinh_p(*args: object, **kwargs: object) -> object:
    """Execute asinh_p."""
    return dispatch("lax", "asinh_p", *args, **kwargs)


def atan2_p(*args: object, **kwargs: object) -> object:
    """Execute atan2_p."""
    return dispatch("lax", "atan2_p", *args, **kwargs)


def atan_p(*args: object, **kwargs: object) -> object:
    """Execute atan_p."""
    return dispatch("lax", "atan_p", *args, **kwargs)


def atanh_p(*args: object, **kwargs: object) -> object:
    """Execute atanh_p."""
    return dispatch("lax", "atanh_p", *args, **kwargs)


def cos_p(*args: object, **kwargs: object) -> object:
    """Execute cos_p."""
    return dispatch("lax", "cos_p", *args, **kwargs)


def cosh_p(*args: object, **kwargs: object) -> object:
    """Execute cosh_p."""
    return dispatch("lax", "cosh_p", *args, **kwargs)


def sin_p(*args: object, **kwargs: object) -> object:
    """Execute sin_p."""
    return dispatch("lax", "sin_p", *args, **kwargs)


def sinh_p(*args: object, **kwargs: object) -> object:
    """Execute sinh_p."""
    return dispatch("lax", "sinh_p", *args, **kwargs)


def tan_p(*args: object, **kwargs: object) -> object:
    """Execute tan_p."""
    return dispatch("lax", "tan_p", *args, **kwargs)


def tanh_p(*args: object, **kwargs: object) -> object:
    """Execute tanh_p."""
    return dispatch("lax", "tanh_p", *args, **kwargs)
