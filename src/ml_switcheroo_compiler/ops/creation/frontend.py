# auto-generate-all

"""Constants & Creation Operations."""

from .frontend_basic import (
    array,
    asarray,
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
from .frontend_windows import bartlett, blackman, hamming, hanning, kaiser

__all__ = [
    "arange",
    "array",
    "asarray",
    "bartlett",
    "blackman",
    "diag",
    "empty",
    "empty_like",
    "eye",
    "full",
    "full_like",
    "hamming",
    "hanning",
    "identity",
    "kaiser",
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
