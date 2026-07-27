# auto-generate-all

"""Constants & Creation Operations."""

from .frontend_basic import (
    array,
    asarray,
    convert_to_numpy,
    convert_to_tensor,
    empty,
    empty_like,
    full,
    full_like,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from .frontend_matrix import diag, eye, identity
from .frontend_random import manual_seed, rand, randint, randn
from .frontend_sequence import arange, linspace

__all__ = [
    "arange",
    "array",
    "asarray",
    "convert_to_numpy",
    "convert_to_tensor",
    "diag",
    "empty",
    "empty_like",
    "eye",
    "full",
    "full_like",
    "identity",
    "linspace",
    "manual_seed",
    "ones",
    "ones_like",
    "rand",
    "randint",
    "randn",
    "zeros",
    "zeros_like",
]
