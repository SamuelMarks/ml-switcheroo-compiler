"""Linear algebra operations package."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.basic import (
    ConvGeneralDilated,
    ConvTranspose,
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
from .decompositions import lu_factor as lu_factor
from .decompositions import norm as norm
from .decompositions import matrix_exponential as matrix_exponential
from .decompositions import matrix_power as matrix_power
from .decompositions import pinv as pinv
from .decompositions import power_iteration as power_iteration
from .decompositions import qr as qr
from .decompositions import slogdet as slogdet
from .decompositions import solve as solve
from .decompositions import tri_inv as tri_inv
from .decompositions import solve_triangular as solve_triangular
from .decompositions import svd as svd
from .fft import fft as fft
from .fft import fft2d as fft2d
from .fft import fft2 as fft2
from .fft import irfft as irfft
from .decompositions import hessenberg as hessenberg
from .decompositions import householder_product as householder_product
from .decompositions import schur as schur
from .decompositions import tridiagonal as tridiagonal
from .decompositions import lu_pivots_to_permutation as lu_pivots_to_permutation

from .fft import fftfreq as fftfreq
from .fft import hfft as hfft
from .fft import ihfft as ihfft
from .fft import rfftfreq as rfftfreq
from .fft import irfft2 as irfft2
from .fft import rfft2 as rfft2
from .fft import fftn as fftn
from .fft import ifftn as ifftn
from .fft import irfftn as irfftn
from .fft import rfftn as rfftn

from .frontend import matrix_norm as matrix_norm
from .frontend import vector_norm as vector_norm
from .frontend import svdvals as svdvals
from .frontend import tensorinv as tensorinv
from .frontend import tensorsolve as tensorsolve
from .frontend import diagonal as diagonal
from .frontend import multi_dot as multi_dot
from .frontend import vecdot as vecdot
from .frontend import trace as trace
from .frontend import matrix_rank as matrix_rank
from .frontend import matrix_transpose as matrix_transpose
from .frontend import sqrtm as sqrtm
from .frontend import adjoint as adjoint
from .frontend import band_part as band_part
from .frontend import cholesky_solve as cholesky_solve
from .frontend import banded_triangular_solve as banded_triangular_solve
from .frontend import eigh_tridiagonal as eigh_tridiagonal

from .frontend import tensor_diag as tensor_diag
from .frontend import tensor_diag_part as tensor_diag_part

from .frontend import (
    LinearOperator as LinearOperator,
    LinearOperatorAdjoint as LinearOperatorAdjoint,
    LinearOperatorBlockDiag as LinearOperatorBlockDiag,
    LinearOperatorBlockLowerTriangular as LinearOperatorBlockLowerTriangular,
    LinearOperatorCirculant as LinearOperatorCirculant,
    LinearOperatorCirculant2D as LinearOperatorCirculant2D,
    LinearOperatorCirculant3D as LinearOperatorCirculant3D,
    LinearOperatorComposition as LinearOperatorComposition,
    LinearOperatorDiag as LinearOperatorDiag,
    LinearOperatorFullMatrix as LinearOperatorFullMatrix,
    LinearOperatorHouseholder as LinearOperatorHouseholder,
    LinearOperatorIdentity as LinearOperatorIdentity,
    LinearOperatorInversion as LinearOperatorInversion,
    LinearOperatorKronecker as LinearOperatorKronecker,
    LinearOperatorLowRankUpdate as LinearOperatorLowRankUpdate,
    LinearOperatorLowerTriangular as LinearOperatorLowerTriangular,
    LinearOperatorPermutation as LinearOperatorPermutation,
    LinearOperatorScaledIdentity as LinearOperatorScaledIdentity,
    LinearOperatorToeplitz as LinearOperatorToeplitz,
    LinearOperatorTridiag as LinearOperatorTridiag,
    LinearOperatorZeros as LinearOperatorZeros,
    conjugate_gradient as conjugate_gradient,
    expm as expm,
    global_norm as global_norm,
    logdet as logdet,
    logm as logm,
    lstsq as lstsq,
    lu as lu,
    lu_matrix_inverse as lu_matrix_inverse,
    lu_reconstruct as lu_reconstruct,
    lu_solve as lu_solve,
    matvec as matvec,
    normalize as normalize,
    set_diag as set_diag,
    triangular_solve as triangular_solve,
    tridiagonal_matmul as tridiagonal_matmul,
    tridiagonal_solve as tridiagonal_solve,
)

from .frontend import diag_part as diag_part
from .basic import Trace as Trace
from .basic import MatrixRank as MatrixRank
from .basic import MatrixTranspose as MatrixTranspose
from .basic import Sqrtm as Sqrtm

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
from .fft import fftnd as fftnd
from .fft import ifftnd as ifftnd
from .fft import rfftnd as rfftnd
from .fft import irfftnd as irfftnd
from .fft import fftshift as fftshift
from .fft import ifftshift as ifftshift

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


@register_op("Vecdot")
class Vecdot(OpDef):
    """Vecdot operator definition."""

    op_name = "Vecdot"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        return args[0] if args else ()


@register_op("CustomLinearSolve")
class CustomLinearSolve(OpDef):
    """CustomLinearSolve operator definition."""

    op_name = "CustomLinearSolve"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        return args[0] if args else ()


from .conv import conv_general_dilated_local as conv_general_dilated_local  # noqa: E402
from .conv import conv_general_dilated_patches as conv_general_dilated_patches  # noqa: E402
from .conv import conv_with_general_padding as conv_with_general_padding  # noqa: E402
from .basic import ConvGeneralDilatedLocal as ConvGeneralDilatedLocal  # noqa: E402
from .basic import ConvGeneralDilatedPatches as ConvGeneralDilatedPatches  # noqa: E402
from .basic import ConvWithGeneralPadding as ConvWithGeneralPadding  # noqa: E402


@register_op("CustomRoot")
class CustomRoot(OpDef):
    """CustomRoot operator definition."""

    op_name = "CustomRoot"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


__all__ = [
    "ConvGeneralDilated",
    "ConvGeneralDilatedLocal",
    "ConvGeneralDilatedPatches",
    "ConvTranspose",
    "ConvWithGeneralPadding",
    "Dot",
    "DotGeneral",
    "Einsum",
    "Fft",
    "LinearOperator",
    "LinearOperatorAdjoint",
    "LinearOperatorBlockDiag",
    "LinearOperatorBlockLowerTriangular",
    "LinearOperatorCirculant",
    "LinearOperatorCirculant2D",
    "LinearOperatorCirculant3D",
    "LinearOperatorComposition",
    "LinearOperatorDiag",
    "LinearOperatorFullMatrix",
    "LinearOperatorHouseholder",
    "LinearOperatorIdentity",
    "LinearOperatorInversion",
    "LinearOperatorKronecker",
    "LinearOperatorLowRankUpdate",
    "LinearOperatorLowerTriangular",
    "LinearOperatorPermutation",
    "LinearOperatorScaledIdentity",
    "LinearOperatorToeplitz",
    "LinearOperatorTridiag",
    "LinearOperatorZeros",
    "Matmul",
    "MatrixRank",
    "MatrixTranspose",
    "Rfft",
    "Sqrtm",
    "Trace",
    "adjoint",
    "band_part",
    "banded_triangular_solve",
    "cholesky",
    "cholesky_solve",
    "conjugate_gradient",
    "conv_general_dilated",
    "conv_general_dilated_local",
    "conv_general_dilated_patches",
    "conv_with_general_padding",
    "convolve",
    "cross",
    "det",
    "diag_part",
    "diagonal",
    "dot",
    "dot_general",
    "eig",
    "eigh",
    "eigh_tridiagonal",
    "eigvalsh",
    "einsum",
    "expm",
    "fft",
    "fft2",
    "fft2d",
    "fft3",
    "fft3d",
    "fftfreq",
    "fftn",
    "fftnd",
    "fftshift",
    "global_norm",
    "hessenberg",
    "hfft",
    "householder_product",
    "ifft",
    "ifft2",
    "ifft2d",
    "ifft3",
    "ifft3d",
    "ifftn",
    "ifftnd",
    "ifftshift",
    "ihfft",
    "inner",
    "inv",
    "irfft",
    "irfft2",
    "irfft2d",
    "irfft3d",
    "irfftn",
    "irfftnd",
    "logdet",
    "logm",
    "lstsq",
    "lu",
    "lu_factor",
    "lu_matrix_inverse",
    "lu_pivots_to_permutation",
    "lu_reconstruct",
    "lu_solve",
    "matmul",
    "matrix_exponential",
    "matrix_norm",
    "matrix_power",
    "matrix_rank",
    "matrix_transpose",
    "matvec",
    "multi_dot",
    "norm",
    "normalize",
    "outer",
    "pinv",
    "power_iteration",
    "qr",
    "rfft",
    "rfft2",
    "rfft2d",
    "rfft3d",
    "rfftfreq",
    "rfftn",
    "rfftnd",
    "schur",
    "set_diag",
    "slogdet",
    "solve",
    "solve_triangular",
    "sqrtm",
    "svd",
    "svdvals",
    "tensor_diag",
    "tensor_diag_part",
    "tensordot",
    "tensorinv",
    "tensorsolve",
    "trace",
    "tri_inv",
    "triangular_solve",
    "tridiagonal",
    "tridiagonal_matmul",
    "tridiagonal_solve",
    "vdot",
    "vecdot",
    "vector_norm",
]
