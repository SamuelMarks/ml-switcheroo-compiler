"""Lax primitives."""

from ml_switcheroo_compiler.core.dispatch import dispatch


def batch_matmul(*args: object, **kwargs: object) -> object:
    """Execute batch_matmul."""
    return dispatch("lax", "batch_matmul", *args, **kwargs)


def custom_linear_solve(*args: object, **kwargs: object) -> object:
    """Execute custom_linear_solve."""
    return dispatch("lax", "custom_linear_solve", *args, **kwargs)


def dot_general_p(*args: object, **kwargs: object) -> object:
    """Execute dot_general_p."""
    return dispatch("lax", "dot_general_p", *args, **kwargs)


def linear_solve_p(*args: object, **kwargs: object) -> object:
    """Execute linear_solve_p."""
    return dispatch("lax", "linear_solve_p", *args, **kwargs)


def pdot(*args: object, **kwargs: object) -> object:
    """Execute pdot."""
    return dispatch("lax", "pdot", *args, **kwargs)


def ragged_dot(*args: object, **kwargs: object) -> object:
    """Execute ragged_dot."""
    return dispatch("lax", "ragged_dot", *args, **kwargs)


def xeinsum(*args: object, **kwargs: object) -> object:
    """Execute xeinsum."""
    return dispatch("lax", "xeinsum", *args, **kwargs)
