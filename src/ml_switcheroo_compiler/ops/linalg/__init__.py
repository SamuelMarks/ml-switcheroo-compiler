# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
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
from ml_switcheroo_compiler.ops.linalg.products import Matmul

from .conv import conv_general_dilated as conv_general_dilated
from .conv import conv_general_dilated_local as conv_general_dilated_local
from .conv import conv_general_dilated_patches as conv_general_dilated_patches
from .conv import conv_with_general_padding as conv_with_general_padding
from .conv_ops import ConvGeneralDilatedLocal as ConvGeneralDilatedLocal
from .conv_ops import ConvGeneralDilatedPatches as ConvGeneralDilatedPatches
from .conv_ops import ConvWithGeneralPadding as ConvWithGeneralPadding
from .decompositions.norms import power_iteration as power_iteration
from .einsum_frontend import einsum as einsum
from .einsum_frontend import tensordot as tensordot
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
from .matrix_ops import cond as cond
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


@register_op("Vecdot")
class Vecdot(OpDef):
    """Vecdot operator definition."""

    op_name: object = "Vecdot"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Evaluate infer_shape operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0] if args else ()


@register_op("CustomLinearSolve")
class CustomLinearSolve(OpDef):
    """CustomLinearSolve operator definition."""

    op_name: object = "CustomLinearSolve"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Evaluate infer_shape operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0] if args else ()


@register_op("CustomRoot")
class CustomRoot(OpDef):
    """CustomRoot operator definition."""

    op_name: object = "CustomRoot"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Evaluate infer_shape operation.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0] if args else ()

    ("ConvGeneralDilated",)
    ("ConvGeneralDilatedLocal",)
    ("ConvGeneralDilatedPatches",)
    ("ConvTranspose",)
    ("ConvWithGeneralPadding",)
    ("CustomLinearSolve",)
    ("CustomRoot",)
    ("Dot",)
    ("DotGeneral",)
    ("Einsum",)
    ("LinearOperator",)
    ("LinearOperatorAdjoint",)
    ("LinearOperatorBlockDiag",)
    ("LinearOperatorBlockLowerTriangular",)
    ("LinearOperatorCirculant",)
    ("LinearOperatorCirculant2D",)
    ("LinearOperatorCirculant3D",)
    ("LinearOperatorComposition",)
    ("LinearOperatorDiag",)
    ("LinearOperatorFullMatrix",)
    ("LinearOperatorHouseholder",)
    ("LinearOperatorIdentity",)
    ("LinearOperatorInversion",)
    ("LinearOperatorKronecker",)
    ("LinearOperatorLowRankUpdate",)
    ("LinearOperatorLowerTriangular",)
    ("LinearOperatorPermutation",)
    ("LinearOperatorScaledIdentity",)
    ("LinearOperatorToeplitz",)
    ("LinearOperatorTridiag",)
    ("LinearOperatorZeros",)
    ("Matmul",)
    ("MatrixRank",)
    ("MatrixTranspose",)
    ("Sqrtm",)
    ("Trace",)
    ("Vecdot",)
    ("addmm",)
    ("adjoint",)
    ("band_part",)
    ("banded_triangular_solve",)
    ("block_masked_mm",)
    ("cholesky",)
    ("cholesky_ex",)
    ("cholesky_solve",)
    ("cond",)
    ("conjugate_gradient",)
    ("conv_general_dilated",)
    ("conv_general_dilated_local",)
    ("conv_general_dilated_patches",)
    ("conv_with_general_padding",)
    ("convolve",)
    ("cross",)
    ("det",)
    ("diag_part",)
    ("diagonal",)
    ("dot",)
    ("dot_general",)
    ("eig",)
    ("eigh",)
    ("eigh_tridiagonal",)
    ("eigvals",)
    ("eigvalsh",)
    ("einsum",)
    ("expm",)
    ("gather_mm",)
    ("global_norm",)
    ("hadamard_transform",)
    ("hessenberg",)
    ("householder_product",)
    ("inner",)
    ("inv",)
    ("inv_ex",)
    ("logdet",)
    ("logm",)
    ("lstsq",)
    ("lu",)
    ("lu_factor",)
    ("lu_matrix_inverse",)
    ("lu_pivots_to_permutation",)
    ("lu_reconstruct",)
    ("lu_solve",)
    ("matmul",)
    ("matrix_exp",)
    ("matrix_exponential",)
    ("matrix_norm",)
    ("matrix_power",)
    ("matrix_rank",)
    ("matrix_transpose",)
    ("matvec",)
    ("multi_dot",)
    ("norm",)
    ("normalize",)
    ("outer",)
    ("pinv",)
    ("power_iteration",)
    ("qdwh",)
    ("qr",)
    ("schur",)
    ("segmented_mm",)
    ("set_diag",)
    ("slogdet",)
    ("solve",)
    ("solve_ex",)
    ("solve_triangular",)
    ("sqrtm",)
    ("svd",)
    ("svdvals",)
    ("tensor_diag",)
    ("tensor_diag_part",)
    ("tensordot",)
    ("tensorinv",)
    ("tensorsolve",)
    ("trace",)
    ("tri_inv",)
    ("triangular_solve",)
    ("tridiagonal",)
    ("tridiagonal_matmul",)
    ("tridiagonal_solve",)
    ("vdot",)
    ("vecdot",)
    ("vector_norm",)


from .dot import Pdot as Pdot

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
    "Pdot",
    "Sqrtm",
    "Trace",
    "Vecdot",
    "addmm",
    "adjoint",
    "band_part",
    "banded_triangular_solve",
    "block_masked_mm",
    "cholesky_solve",
    "cond",
    "conjugate_gradient",
    "conv_general_dilated",
    "conv_general_dilated_local",
    "conv_general_dilated_patches",
    "conv_with_general_padding",
    "convolve",
    "cross",
    "diag_part",
    "diagonal",
    "dot",
    "dot_general",
    "eigh_tridiagonal",
    "einsum",
    "expm",
    "gather_mm",
    "global_norm",
    "hadamard_transform",
    "inner",
    "logdet",
    "logm",
    "lstsq",
    "lu",
    "lu_matrix_inverse",
    "lu_reconstruct",
    "lu_solve",
    "matmul",
    "matrix_norm",
    "matrix_rank",
    "matrix_transpose",
    "matvec",
    "multi_dot",
    "normalize",
    "outer",
    "power_iteration",
    "segmented_mm",
    "set_diag",
    "sqrtm",
    "svdvals",
    "tensor_diag",
    "tensor_diag_part",
    "tensordot",
    "tensorinv",
    "tensorsolve",
    "trace",
    "triangular_solve",
    "tridiagonal_matmul",
    "tridiagonal_solve",
    "vdot",
    "vecdot",
    "vector_norm",
]
