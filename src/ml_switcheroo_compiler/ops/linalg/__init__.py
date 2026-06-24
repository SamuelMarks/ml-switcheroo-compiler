"""Linear algebra operations package."""

from ml_switcheroo_compiler.ops.linalg.basic import (
    ConvGeneralDilated,
    Fft,
    Matmul,
    Rfft,
)
from ml_switcheroo_compiler.ops.linalg.dot import (
    Dot,
    DotGeneral,
)
from ml_switcheroo_compiler.ops.linalg.einsum import (
    Einsum,
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
from .fft import fft2d as fft2d
from .fft import fft2 as fft2
from .fft import irfft as irfft
from .fft import ifft as ifft
from .fft import ifft2d as ifft2d
from .fft import ifft2 as ifft2
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
    "eig",
    "eigh",
    "eigvalsh",
    "einsum",
    "fft",
    "fft2d",
    "fft2",
    "ifft",
    "ifft2d",
    "ifft2",
    "irfft",
    "inner",
    "inv",
    "logdet",
    "lstsq",
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


def eig(input: object) -> tuple[object, object]:
    """Computes eigenvalues and eigenvectors of a square matrix."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    w, v = backend.execute_op("Eig", getattr(input, "data", input))
    return w, v


def logdet(input: object) -> object:
    """Computes log of the determinant."""
    from ml_switcheroo_compiler.ops.linalg.decompositions import slogdet

    sign, ldet = slogdet(input)
    # Usually logdet is only defined for positive determinant, but we just return ldet
    return ldet


def lstsq(a: object, b: object, rcond: float = 1e-15) -> object:
    """Returns the least-squares solution to a linear matrix equation."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    res = backend.execute_op("Lstsq", getattr(a, "data", a), getattr(b, "data", b), rcond=rcond)
    # The return value might be a tuple (x, residuals, rank, s) depending on backend
    return res
