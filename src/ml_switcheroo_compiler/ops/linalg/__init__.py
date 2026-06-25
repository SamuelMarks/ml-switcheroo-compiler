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
from .decompositions import lu_solve as lu_solve
from .decompositions import norm as norm
from .decompositions import matrix_exponential as matrix_exponential
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
from .fft import fft3d as fft3d
from .fft import fft3 as fft3
from .fft import ifft3d as ifft3d
from .fft import ifft3 as ifft3
from .fft import rfft2d as rfft2d
from .fft import rfft3d as rfft3d
from .fft import irfft2d as irfft2d
from .fft import irfft3d as irfft3d
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


__all__ = [
    "ConvGeneralDilated",
    "Dot",
    "DotGeneral",
    "Einsum",
    "Fft",
    "Matmul",
    "Rfft",
    "cholesky",
    "conv_general_dilated",
    "convolve",
    "cross",
    "det",
    "dot",
    "dot_general",
    "eig",
    "eigh",
    "eigvalsh",
    "einsum",
    "fft",
    "fft2",
    "fft2d",
    "fft3",
    "fft3d",
    "ifft",
    "ifft2",
    "ifft2d",
    "ifft3",
    "ifft3d",
    "inner",
    "inv",
    "irfft",
    "irfft2d",
    "irfft3d",
    "logdet",
    "lstsq",
    "lu",
    "lu_factor",
    "lu_solve",
    "matmul",
    "matrix_exponential",
    "matrix_power",
    "norm",
    "outer",
    "pinv",
    "power_iteration",
    "qr",
    "rfft",
    "rfft2d",
    "rfft3d",
    "slogdet",
    "solve",
    "solve_triangular",
    "svd",
    "tensordot",
    "vdot",
]
