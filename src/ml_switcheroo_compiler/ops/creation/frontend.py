# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module frontend.py."""


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
