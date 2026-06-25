"""Creation operations package."""

from ml_switcheroo_compiler.ops.creation.basic import Arange, CreationOp, Full, Ones, Zeros

from .frontend import arange as arange
from .frontend import array as array
from .frontend import asarray as asarray
from .frontend import bartlett as bartlett
from .frontend import blackman as blackman
from .frontend import diag as diag
from .frontend import empty as empty
from .frontend import empty_like as empty_like
from .frontend import eye as eye
from .frontend import full as full
from .frontend import full_like as full_like
from .frontend import hamming as hamming
from .frontend import hanning as hanning
from .frontend import identity as identity
from .frontend import kaiser as kaiser
from .frontend import linspace as linspace
from .frontend import manual_seed as manual_seed
from .frontend import ones as ones
from .frontend import ones_like as ones_like
from .frontend import rand as rand
from .frontend import randint as randint
from .frontend import randn as randn
from .frontend import zeros as zeros
from .frontend import zeros_like as zeros_like

__all__ = [
    "Arange",
    "CreationOp",
    "Full",
    "Ones",
    "Zeros",
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
