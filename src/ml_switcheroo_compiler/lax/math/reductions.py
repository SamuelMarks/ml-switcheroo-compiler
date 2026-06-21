"""Reductions operations."""

from ml_switcheroo_compiler.core.dispatch import dispatch


def approx_max_k(*args: object, **kwargs: object) -> object:
    """Execute approx_max_k."""
    return dispatch("lax", "approx_max_k", *args, **kwargs)


def approx_min_k(*args: object, **kwargs: object) -> object:
    """Execute approx_min_k."""
    return dispatch("lax", "approx_min_k", *args, **kwargs)


def approx_top_k_p(*args: object, **kwargs: object) -> object:
    """Execute approx_top_k_p."""
    return dispatch("lax", "approx_top_k_p", *args, **kwargs)


def clamp_p(*args: object, **kwargs: object) -> object:
    """Execute clamp_p."""
    return dispatch("lax", "clamp_p", *args, **kwargs)


def cumlogsumexp(*args: object, **kwargs: object) -> object:
    """Execute cumlogsumexp."""
    return dispatch("lax", "cumlogsumexp", *args, **kwargs)


def cumlogsumexp_p(*args: object, **kwargs: object) -> object:
    """Execute cumlogsumexp_p."""
    return dispatch("lax", "cumlogsumexp_p", *args, **kwargs)


def cummax(*args: object, **kwargs: object) -> object:
    """Execute cummax."""
    return dispatch("lax", "cummax", *args, **kwargs)


def cummax_p(*args: object, **kwargs: object) -> object:
    """Execute cummax_p."""
    return dispatch("lax", "cummax_p", *args, **kwargs)


def cummin(*args: object, **kwargs: object) -> object:
    """Execute cummin."""
    return dispatch("lax", "cummin", *args, **kwargs)


def cummin_p(*args: object, **kwargs: object) -> object:
    """Execute cummin_p."""
    return dispatch("lax", "cummin_p", *args, **kwargs)


def cumprod(*args: object, **kwargs: object) -> object:
    """Execute cumprod."""
    return dispatch("lax", "cumprod", *args, **kwargs)


def cumprod_p(*args: object, **kwargs: object) -> object:
    """Execute cumprod_p."""
    return dispatch("lax", "cumprod_p", *args, **kwargs)


def cumsum_p(*args: object, **kwargs: object) -> object:
    """Execute cumsum_p."""
    return dispatch("lax", "cumsum_p", *args, **kwargs)


def max_p(*args: object, **kwargs: object) -> object:
    """Execute max_p."""
    return dispatch("lax", "max_p", *args, **kwargs)


def min_p(*args: object, **kwargs: object) -> object:
    """Execute min_p."""
    return dispatch("lax", "min_p", *args, **kwargs)


def reduce_and_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_and_p."""
    return dispatch("lax", "reduce_and_p", *args, **kwargs)


def reduce_max_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_max_p."""
    return dispatch("lax", "reduce_max_p", *args, **kwargs)


def reduce_min_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_min_p."""
    return dispatch("lax", "reduce_min_p", *args, **kwargs)


def reduce_or_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_or_p."""
    return dispatch("lax", "reduce_or_p", *args, **kwargs)


def reduce_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_p."""
    return dispatch("lax", "reduce_p", *args, **kwargs)


def reduce_prod_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_prod_p."""
    return dispatch("lax", "reduce_prod_p", *args, **kwargs)


def reduce_sum_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_sum_p."""
    return dispatch("lax", "reduce_sum_p", *args, **kwargs)


def reduce_xor_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_xor_p."""
    return dispatch("lax", "reduce_xor_p", *args, **kwargs)


def scatter_add_p(*args: object, **kwargs: object) -> object:
    """Execute scatter_add_p."""
    return dispatch("lax", "scatter_add_p", *args, **kwargs)


def scatter_max(*args: object, **kwargs: object) -> object:
    """Execute scatter_max."""
    return dispatch("lax", "scatter_max", *args, **kwargs)


def scatter_max_p(*args: object, **kwargs: object) -> object:
    """Execute scatter_max_p."""
    return dispatch("lax", "scatter_max_p", *args, **kwargs)


def scatter_min(*args: object, **kwargs: object) -> object:
    """Execute scatter_min."""
    return dispatch("lax", "scatter_min", *args, **kwargs)


def scatter_min_p(*args: object, **kwargs: object) -> object:
    """Execute scatter_min_p."""
    return dispatch("lax", "scatter_min_p", *args, **kwargs)


def scatter_mul(*args: object, **kwargs: object) -> object:
    """Execute scatter_mul."""
    return dispatch("lax", "scatter_mul", *args, **kwargs)


def scatter_mul_p(*args: object, **kwargs: object) -> object:
    """Execute scatter_mul_p."""
    return dispatch("lax", "scatter_mul_p", *args, **kwargs)


def select_and_scatter_add_p(*args: object, **kwargs: object) -> object:
    """Execute select_and_scatter_add_p."""
    return dispatch("lax", "select_and_scatter_add_p", *args, **kwargs)


def select_and_scatter_p(*args: object, **kwargs: object) -> object:
    """Execute select_and_scatter_p."""
    return dispatch("lax", "select_and_scatter_p", *args, **kwargs)
