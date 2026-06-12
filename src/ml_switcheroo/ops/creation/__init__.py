"""Creation operations package."""

from ml_switcheroo.ops.creation.basic import (
    CreationOp,
    Zeros,
    Ones,
    Full,
    Arange,
)

__all__ = [
    "CreationOp",
    "Zeros",
    "Ones",
    "Full",
    "Arange",
]
from .frontend import (
    arange as arange,
    diag as diag,
    empty as empty,
    eye as eye,
    full as full,
    full_like as full_like,
    identity as identity,
    linspace as linspace,
    ones as ones,
    ones_like as ones_like,
    zeros as zeros,
    zeros_like as zeros_like,
)
