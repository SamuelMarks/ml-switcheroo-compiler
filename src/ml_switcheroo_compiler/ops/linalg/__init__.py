# auto-generate-all
# exclude_exports: OpDef, register_op

"""Linear algebra operations package."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.conv_ops import ConvGeneralDilated, ConvTranspose
from ml_switcheroo_compiler.ops.linalg.dot import (
    Dot,
    DotGeneral,
)
from ml_switcheroo_compiler.ops.linalg.einsum import (
    Einsum,
)
from ml_switcheroo_compiler.ops.linalg.fft_ops import Fft, Rfft
from ml_switcheroo_compiler.ops.linalg.products import Matmul

from .conv import conv_general_dilated as conv_general_dilated
from .conv import conv_general_dilated_local as conv_general_dilated_local
from .conv import conv_general_dilated_patches as conv_general_dilated_patches
from .conv import conv_with_general_padding as conv_with_general_padding
from .conv_ops import ConvGeneralDilatedLocal as ConvGeneralDilatedLocal
from .conv_ops import ConvGeneralDilatedPatches as ConvGeneralDilatedPatches
from .conv_ops import ConvWithGeneralPadding as ConvWithGeneralPadding
from .decompositions import cholesky as cholesky
from .decompositions import det as det
from .decompositions import eigh as eigh
from .decompositions import eigvalsh as eigvalsh
from .decompositions import hessenberg as hessenberg
from .decompositions import householder_product as householder_product
from .decompositions import inv as inv
from .decompositions import lu_factor as lu_factor
from .decompositions import lu_pivots_to_permutation as lu_pivots_to_permutation
from .decompositions import matrix_exponential as matrix_exponential
from .decompositions import matrix_power as matrix_power
from .decompositions import norm as norm
from .decompositions import pinv as pinv
from .decompositions import power_iteration as power_iteration
from .decompositions import qr as qr
from .decompositions import schur as schur
from .decompositions import slogdet as slogdet
from .decompositions import solve as solve
from .decompositions import solve_triangular as solve_triangular
from .decompositions import svd as svd
from .decompositions import tri_inv as tri_inv
from .decompositions import tridiagonal as tridiagonal
from .einsum_frontend import einsum as einsum
from .einsum_frontend import tensordot as tensordot
from .fft import fft as fft
from .fft import fft2 as fft2
from .fft import fft2d as fft2d
from .fft import fft3 as fft3
from .fft import fft3d as fft3d
from .fft import fftfreq as fftfreq
from .fft import fftn as fftn
from .fft import fftnd as fftnd
from .fft import fftshift as fftshift
from .fft import hfft as hfft
from .fft import ifft as ifft
from .fft import ifft2 as ifft2
from .fft import ifft2d as ifft2d
from .fft import ifft3 as ifft3
from .fft import ifft3d as ifft3d
from .fft import ifftn as ifftn
from .fft import ifftnd as ifftnd
from .fft import ifftshift as ifftshift
from .fft import ihfft as ihfft
from .fft import irfft as irfft
from .fft import irfft2 as irfft2
from .fft import irfft2d as irfft2d
from .fft import irfft3d as irfft3d
from .fft import irfftn as irfftn
from .fft import irfftnd as irfftnd
from .fft import rfft as rfft
from .fft import rfft2 as rfft2
from .fft import rfft2d as rfft2d
from .fft import rfft3d as rfft3d
from .fft import rfftfreq as rfftfreq
from .fft import rfftn as rfftn
from .fft import rfftnd as rfftnd
from .linear_operator import LinearOperator as LinearOperator
from .linear_operator import LinearOperatorAdjoint as LinearOperatorAdjoint
from .linear_operator import LinearOperatorBlockDiag as LinearOperatorBlockDiag
from .linear_operator import (
    LinearOperatorBlockLowerTriangular as LinearOperatorBlockLowerTriangular,
)
from .linear_operator import LinearOperatorCirculant as LinearOperatorCirculant
from .linear_operator import LinearOperatorCirculant2D as LinearOperatorCirculant2D
from .linear_operator import LinearOperatorCirculant3D as LinearOperatorCirculant3D
from .linear_operator import LinearOperatorComposition as LinearOperatorComposition
from .linear_operator import LinearOperatorDiag as LinearOperatorDiag
from .linear_operator import LinearOperatorFullMatrix as LinearOperatorFullMatrix
from .linear_operator import LinearOperatorHouseholder as LinearOperatorHouseholder
from .linear_operator import LinearOperatorIdentity as LinearOperatorIdentity
from .linear_operator import LinearOperatorInversion as LinearOperatorInversion
from .linear_operator import LinearOperatorKronecker as LinearOperatorKronecker
from .linear_operator import LinearOperatorLowerTriangular as LinearOperatorLowerTriangular
from .linear_operator import LinearOperatorLowRankUpdate as LinearOperatorLowRankUpdate
from .linear_operator import LinearOperatorPermutation as LinearOperatorPermutation
from .linear_operator import LinearOperatorScaledIdentity as LinearOperatorScaledIdentity
from .linear_operator import LinearOperatorToeplitz as LinearOperatorToeplitz
from .linear_operator import LinearOperatorTridiag as LinearOperatorTridiag
from .linear_operator import LinearOperatorZeros as LinearOperatorZeros
from .matmul import addmm as addmm
from .matmul import block_masked_mm as block_masked_mm
from .matmul import convolve as convolve
from .matmul import dot as dot
from .matmul import dot_general as dot_general
from .matmul import gather_mm as gather_mm
from .matmul import inner as inner
from .matmul import matmul as matmul
from .matmul import matvec as matvec
from .matmul import multi_dot as multi_dot
from .matmul import outer as outer
from .matmul import segmented_mm as segmented_mm
from .matmul import vdot as vdot
from .matmul import vecdot as vecdot
from .matrix_ops import adjoint as adjoint
from .matrix_ops import band_part as band_part
from .matrix_ops import cross as cross
from .matrix_ops import diag_part as diag_part
from .matrix_ops import diagonal as diagonal
from .matrix_ops import eigh_tridiagonal as eigh_tridiagonal
from .matrix_ops import expm as expm
from .matrix_ops import global_norm as global_norm
from .matrix_ops import logdet as logdet
from .matrix_ops import logm as logm
from .matrix_ops import matrix_norm as matrix_norm
from .matrix_ops import matrix_rank as matrix_rank
from .matrix_ops import matrix_transpose as matrix_transpose
from .matrix_ops import normalize as normalize
from .matrix_ops import set_diag as set_diag
from .matrix_ops import sqrtm as sqrtm
from .matrix_ops import svdvals as svdvals
from .matrix_ops import tensor_diag as tensor_diag
from .matrix_ops import tensor_diag_part as tensor_diag_part
from .matrix_ops import trace as trace
from .matrix_ops import tridiagonal_matmul as tridiagonal_matmul
from .matrix_ops import vector_norm as vector_norm
from .products import MatrixRank as MatrixRank
from .products import MatrixTranspose as MatrixTranspose
from .products import Trace as Trace
from .solvers import Sqrtm as Sqrtm
from .solvers import banded_triangular_solve as banded_triangular_solve
from .solvers import cholesky_solve as cholesky_solve
from .solvers import conjugate_gradient as conjugate_gradient
from .solvers import lstsq as lstsq
from .solvers import lu as lu
from .solvers import lu_matrix_inverse as lu_matrix_inverse
from .solvers import lu_reconstruct as lu_reconstruct
from .solvers import lu_solve as lu_solve
from .solvers import tensorinv as tensorinv
from .solvers import tensorsolve as tensorsolve
from .solvers import triangular_solve as triangular_solve
from .solvers import tridiagonal_solve as tridiagonal_solve
from .transform import hadamard_transform as hadamard_transform


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
        """Function docstring."""
        return args[0] if args else ()


@register_op("CustomLinearSolve")
class CustomLinearSolve(OpDef):
    """CustomLinearSolve operator definition."""

    op_name = "CustomLinearSolve"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Function docstring."""
        return args[0] if args else ()


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
    "CustomLinearSolve",
    "CustomRoot",
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
    "Vecdot",
    "addmm",
    "adjoint",
    "band_part",
    "banded_triangular_solve",
    "block_masked_mm",
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
    "gather_mm",
    "global_norm",
    "hadamard_transform",
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
    "segmented_mm",
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
