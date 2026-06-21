"""Arithmetic operations."""

from ml_switcheroo_compiler.core.dispatch import dispatch


def abs_p(*args: object, **kwargs: object) -> object:
    """Execute abs_p."""
    return dispatch("lax", "abs_p", *args, **kwargs)


def add_p(*args: object, **kwargs: object) -> object:
    """Execute add_p."""
    return dispatch("lax", "add_p", *args, **kwargs)


def cbrt_p(*args: object, **kwargs: object) -> object:
    """Execute cbrt_p."""
    return dispatch("lax", "cbrt_p", *args, **kwargs)


def ceil_p(*args: object, **kwargs: object) -> object:
    """Execute ceil_p."""
    return dispatch("lax", "ceil_p", *args, **kwargs)


def complex(*args: object, **kwargs: object) -> object:
    """Execute complex."""
    return dispatch("lax", "complex", *args, **kwargs)


def complex_p(*args: object, **kwargs: object) -> object:
    """Execute complex_p."""
    return dispatch("lax", "complex_p", *args, **kwargs)


def conj_p(*args: object, **kwargs: object) -> object:
    """Execute conj_p."""
    return dispatch("lax", "conj_p", *args, **kwargs)


def div_p(*args: object, **kwargs: object) -> object:
    """Execute div_p."""
    return dispatch("lax", "div_p", *args, **kwargs)


def floor_p(*args: object, **kwargs: object) -> object:
    """Execute floor_p."""
    return dispatch("lax", "floor_p", *args, **kwargs)


def imag_p(*args: object, **kwargs: object) -> object:
    """Execute imag_p."""
    return dispatch("lax", "imag_p", *args, **kwargs)


def integer_pow(*args: object, **kwargs: object) -> object:
    """Execute integer_pow."""
    return dispatch("lax", "integer_pow", *args, **kwargs)


def integer_pow_p(*args: object, **kwargs: object) -> object:
    """Execute integer_pow_p."""
    return dispatch("lax", "integer_pow_p", *args, **kwargs)


def mul_p(*args: object, **kwargs: object) -> object:
    """Execute mul_p."""
    return dispatch("lax", "mul_p", *args, **kwargs)


def neg(*args: object, **kwargs: object) -> object:
    """Execute neg."""
    return dispatch("lax", "neg", *args, **kwargs)


def neg_p(*args: object, **kwargs: object) -> object:
    """Execute neg_p."""
    return dispatch("lax", "neg_p", *args, **kwargs)


def pow(*args: object, **kwargs: object) -> object:
    """Execute pow."""
    return dispatch("lax", "pow", *args, **kwargs)


def pow_p(*args: object, **kwargs: object) -> object:
    """Execute pow_p."""
    return dispatch("lax", "pow_p", *args, **kwargs)


def real_p(*args: object, **kwargs: object) -> object:
    """Execute real_p."""
    return dispatch("lax", "real_p", *args, **kwargs)


def reduce_precision(*args: object, **kwargs: object) -> object:
    """Execute reduce_precision."""
    return dispatch("lax", "reduce_precision", *args, **kwargs)


def reduce_precision_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_precision_p."""
    return dispatch("lax", "reduce_precision_p", *args, **kwargs)


def rem(*args: object, **kwargs: object) -> object:
    """Execute rem."""
    return dispatch("lax", "rem", *args, **kwargs)


def rem_p(*args: object, **kwargs: object) -> object:
    """Execute rem_p."""
    return dispatch("lax", "rem_p", *args, **kwargs)


def round_p(*args: object, **kwargs: object) -> object:
    """Execute round_p."""
    return dispatch("lax", "round_p", *args, **kwargs)


def sign_p(*args: object, **kwargs: object) -> object:
    """Execute sign_p."""
    return dispatch("lax", "sign_p", *args, **kwargs)


def sub_p(*args: object, **kwargs: object) -> object:
    """Execute sub_p."""
    return dispatch("lax", "sub_p", *args, **kwargs)
