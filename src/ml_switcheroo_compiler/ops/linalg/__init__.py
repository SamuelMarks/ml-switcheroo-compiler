"""Linear algebra operations package."""

from ml_switcheroo_compiler.ops.linalg.basic import (
    ConvGeneralDilated,
    Dot,
    DotGeneral,
    Einsum,
    Fft,
    Matmul,
    Rfft,
)

from .conv import conv_general_dilated as conv_general_dilated
from .decompositions import cholesky as cholesky
from .decompositions import det as det
from .decompositions import eigh as eigh
from .decompositions import eigvalsh as eigvalsh
from .decompositions import inv as inv
from .decompositions import lu as lu
from .decompositions import lu_factor as lu_factor
from .decompositions import matrix_power as matrix_power
from .decompositions import pinv as pinv
from .decompositions import power_iteration as power_iteration
from .decompositions import qr as qr
from .decompositions import slogdet as slogdet
from .decompositions import solve as solve
from .decompositions import solve_triangular as solve_triangular
from .decompositions import svd as svd
from .fft import fft as fft
from .fft import rfft as rfft
from .frontend import convolve as convolve
from .frontend import cross as cross
from .frontend import dot as dot
from .frontend import dot_general as dot_general
from .frontend import einsum as einsum
from .frontend import inner as inner
from .frontend import matmul as matmul
from .frontend import outer as outer
from .frontend import tensordot as tensordot
from .frontend import vdot as vdot

__all__ = [
    "ConvGeneralDilated",
    "Dot",
    "DotGeneral",
    "Einsum",
    "Fft",
    "Matmul",
    "Rfft",
    "cholesky",
    "convolve",
    "conv_general_dilated",
    "cross",
    "det",
    "dot",
    "dot_general",
    "eigh",
    "eigvalsh",
    "einsum",
    "fft",
    "inner",
    "inv",
    "lu",
    "lu_factor",
    "matmul",
    "matrix_power",
    "outer",
    "pinv",
    "power_iteration",
    "qr",
    "rfft",
    "slogdet",
    "solve",
    "solve_triangular",
    "svd",
    "tensordot",
    "vdot",
]
