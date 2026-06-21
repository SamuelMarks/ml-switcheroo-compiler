"""Random operations."""

from ml_switcheroo_compiler.core.dispatch import dispatch


def rng_bit_generator(*args: object, **kwargs: object) -> object:
    """Execute rng_bit_generator."""
    return dispatch("lax", "rng_bit_generator", *args, **kwargs)


def rng_bit_generator_p(*args: object, **kwargs: object) -> object:
    """Execute rng_bit_generator_p."""
    return dispatch("lax", "rng_bit_generator_p", *args, **kwargs)


def rng_uniform(*args: object, **kwargs: object) -> object:
    """Execute rng_uniform."""
    return dispatch("lax", "rng_uniform", *args, **kwargs)


def rng_uniform_p(*args: object, **kwargs: object) -> object:
    """Execute rng_uniform_p."""
    return dispatch("lax", "rng_uniform_p", *args, **kwargs)
