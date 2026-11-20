"""Creation ops."""

from __future__ import annotations

from .api_basic import (
    array,
    asarray,
    convert_to_numpy,
    convert_to_tensor,
    empty,
    empty_like,
    frombuffer,
    full,
    full_like,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from .api_matrix import diag, eye, identity
from .api_random import manual_seed, rand, randint, randn
from .api_sequence import arange, linspace

__all__ = [
    "eye",
    "identity",
    "diag",
    "rand",
    "randn",
    "randint",
    "manual_seed",
    "arange",
    "linspace",
    "array",
    "asarray",
    "convert_to_tensor",
    "zeros",
    "ones",
    "full",
    "zeros_like",
    "ones_like",
    "full_like",
    "empty",
    "frombuffer",
    "empty_like",
    "convert_to_numpy",
]
