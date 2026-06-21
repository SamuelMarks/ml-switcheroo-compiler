"""Lax primitives."""

from ml_switcheroo_compiler.core.dispatch import dispatch


def after_all(*args: object, **kwargs: object) -> object:
    """Execute after_all."""
    return dispatch("lax", "after_all", *args, **kwargs)


def after_all_p(*args: object, **kwargs: object) -> object:
    """Execute after_all_p."""
    return dispatch("lax", "after_all_p", *args, **kwargs)


def all_gather(*args: object, **kwargs: object) -> object:
    """Execute all_gather."""
    return dispatch("lax", "all_gather", *args, **kwargs)


def all_gather_p(*args: object, **kwargs: object) -> object:
    """Execute all_gather_p."""
    return dispatch("lax", "all_gather_p", *args, **kwargs)


def all_to_all(*args: object, **kwargs: object) -> object:
    """Execute all_to_all."""
    return dispatch("lax", "all_to_all", *args, **kwargs)


def all_to_all_p(*args: object, **kwargs: object) -> object:
    """Execute all_to_all_p."""
    return dispatch("lax", "all_to_all_p", *args, **kwargs)


def gather_p(*args: object, **kwargs: object) -> object:
    """Execute gather_p."""
    return dispatch("lax", "gather_p", *args, **kwargs)


def padtype_to_pads(*args: object, **kwargs: object) -> object:
    """Execute padtype_to_pads."""
    return dispatch("lax", "padtype_to_pads", *args, **kwargs)


def pbroadcast(*args: object, **kwargs: object) -> object:
    """Execute pbroadcast."""
    return dispatch("lax", "pbroadcast", *args, **kwargs)


def pmax(*args: object, **kwargs: object) -> object:
    """Execute pmax."""
    return dispatch("lax", "pmax", *args, **kwargs)


def pmax_p(*args: object, **kwargs: object) -> object:
    """Execute pmax_p."""
    return dispatch("lax", "pmax_p", *args, **kwargs)


def pmin(*args: object, **kwargs: object) -> object:
    """Execute pmin."""
    return dispatch("lax", "pmin", *args, **kwargs)


def pmin_p(*args: object, **kwargs: object) -> object:
    """Execute pmin_p."""
    return dispatch("lax", "pmin_p", *args, **kwargs)


def ppermute(*args: object, **kwargs: object) -> object:
    """Execute ppermute."""
    return dispatch("lax", "ppermute", *args, **kwargs)


def ppermute_p(*args: object, **kwargs: object) -> object:
    """Execute ppermute_p."""
    return dispatch("lax", "ppermute_p", *args, **kwargs)


def pshuffle(*args: object, **kwargs: object) -> object:
    """Execute pshuffle."""
    return dispatch("lax", "pshuffle", *args, **kwargs)


def psum_p(*args: object, **kwargs: object) -> object:
    """Execute psum_p."""
    return dispatch("lax", "psum_p", *args, **kwargs)


def psum_scatter(*args: object, **kwargs: object) -> object:
    """Execute psum_scatter."""
    return dispatch("lax", "psum_scatter", *args, **kwargs)


def pswapaxes(*args: object, **kwargs: object) -> object:
    """Execute pswapaxes."""
    return dispatch("lax", "pswapaxes", *args, **kwargs)


def select_and_gather_add_p(*args: object, **kwargs: object) -> object:
    """Execute select_and_gather_add_p."""
    return dispatch("lax", "select_and_gather_add_p", *args, **kwargs)


def sharding_constraint_p(*args: object, **kwargs: object) -> object:
    """Execute sharding_constraint_p."""
    return dispatch("lax", "sharding_constraint_p", *args, **kwargs)


def with_sharding_constraint(*args: object, **kwargs: object) -> object:
    """Execute with_sharding_constraint."""
    return dispatch("lax", "with_sharding_constraint", *args, **kwargs)
