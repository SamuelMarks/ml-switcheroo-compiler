# ruff: noqa: E402, E501, F403
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

    op_name = "Vecdot"

    def infer_shape(self, *args, **kwargs):
        """Evaluate infer_shape operation contracting vector dimension.

        Args:
            *args (Any): Positional args (x, y).
            **kwargs (Any): Keyword args (axis).

        Returns:
            tuple[int, ...]: Resulting shape after contracting vector dimension.
        """
        if not args or len(args) < 2:
            return ()
        x_shape = getattr(args[0], "shape", ())
        y_shape = getattr(args[1], "shape", ())
        if not x_shape or not y_shape:
            return ()
        axis = int(kwargs.get("axis", -1))
        x_norm = axis if axis >= 0 else len(x_shape) + axis
        y_norm = axis if axis >= 0 else len(y_shape) + axis
        x_batch = tuple(dim for idx, dim in enumerate(x_shape) if idx != x_norm)
        y_batch = tuple(dim for idx, dim in enumerate(y_shape) if idx != y_norm)
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        return broadcast_shapes(x_batch, y_batch)


@register_op("CustomLinearSolve")
class CustomLinearSolve(OpDef):
    """CustomLinearSolve operator definition."""

    op_name = "CustomLinearSolve"

    def infer_shape(self, *args, **kwargs):
        """Evaluate infer_shape operation solving AX = B.

        Args:
            *args (Any): Positional args (A, B).
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Resulting shape matching broadcasted solution X.
        """
        if not args or len(args) < 2:
            return ()
        a_shape = getattr(args[0], "shape", ())
        b_shape = getattr(args[1], "shape", ())
        if len(a_shape) < 2 or len(b_shape) < 1:
            return ()
        m = a_shape[-1]
        batch_a = a_shape[:-2]
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        if len(b_shape) == len(a_shape) - 1:
            batch_b = b_shape[:-1]
            return broadcast_shapes(batch_a, batch_b) + (m,)
        batch_b = b_shape[:-2]
        k = b_shape[-1]
        return broadcast_shapes(batch_a, batch_b) + (m, k)


@register_op("CustomRoot")
class CustomRoot(OpDef):
    """CustomRoot operator definition."""

    op_name = "CustomRoot"

    def infer_shape(self, *args, **kwargs):
        """Evaluate infer_shape operation for root finding.

        Args:
            *args (Any): Positional args (fn, x0).
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Shape matching initial guess x0.
        """
        if len(args) > 1 and hasattr(args[1], "shape"):
            return tuple(args[1].shape)
        if args and hasattr(args[0], "shape"):
            return tuple(args[0].shape)
        return ()

    ("ConvGeneralDilated",)  # noqa: B018
    ("ConvGeneralDilatedLocal",)  # noqa: B018
    ("ConvGeneralDilatedPatches",)  # noqa: B018
    ("ConvTranspose",)  # noqa: B018
    ("ConvWithGeneralPadding",)  # noqa: B018
    ("CustomLinearSolve",)  # noqa: B018
    ("CustomRoot",)  # noqa: B018
    ("Dot",)  # noqa: B018
    ("DotGeneral",)  # noqa: B018
    ("Einsum",)  # noqa: B018
    ("LinearOperator",)  # noqa: B018
    ("LinearOperatorAdjoint",)  # noqa: B018
    ("LinearOperatorBlockDiag",)  # noqa: B018
    ("LinearOperatorBlockLowerTriangular",)  # noqa: B018
    ("LinearOperatorCirculant",)  # noqa: B018
    ("LinearOperatorCirculant2D",)  # noqa: B018
    ("LinearOperatorCirculant3D",)  # noqa: B018
    ("LinearOperatorComposition",)  # noqa: B018
    ("LinearOperatorDiag",)  # noqa: B018
    ("LinearOperatorFullMatrix",)  # noqa: B018
    ("LinearOperatorHouseholder",)  # noqa: B018
    ("LinearOperatorIdentity",)  # noqa: B018
    ("LinearOperatorInversion",)  # noqa: B018
    ("LinearOperatorKronecker",)  # noqa: B018
    ("LinearOperatorLowRankUpdate",)  # noqa: B018
    ("LinearOperatorLowerTriangular",)  # noqa: B018
    ("LinearOperatorPermutation",)  # noqa: B018
    ("LinearOperatorScaledIdentity",)  # noqa: B018
    ("LinearOperatorToeplitz",)  # noqa: B018
    ("LinearOperatorTridiag",)  # noqa: B018
    ("LinearOperatorZeros",)  # noqa: B018
    ("Matmul",)  # noqa: B018
    ("MatrixRank",)  # noqa: B018
    ("MatrixTranspose",)  # noqa: B018
    ("Sqrtm",)  # noqa: B018
    ("Trace",)  # noqa: B018
    ("Vecdot",)  # noqa: B018
    ("addmm",)  # noqa: B018
    ("adjoint",)  # noqa: B018
    ("band_part",)  # noqa: B018
    ("banded_triangular_solve",)  # noqa: B018
    ("block_masked_mm",)  # noqa: B018
    ("cholesky",)  # noqa: B018
    ("cholesky_ex",)  # noqa: B018
    ("cholesky_solve",)  # noqa: B018
    ("cond",)  # noqa: B018
    ("conjugate_gradient",)  # noqa: B018
    ("conv_general_dilated",)  # noqa: B018
    ("conv_general_dilated_local",)  # noqa: B018
    ("conv_general_dilated_patches",)  # noqa: B018
    ("conv_with_general_padding",)  # noqa: B018
    ("convolve",)  # noqa: B018
    ("cross",)  # noqa: B018
    ("det",)  # noqa: B018
    ("diag_part",)  # noqa: B018
    ("diagonal",)  # noqa: B018
    ("dot",)  # noqa: B018
    ("dot_general",)  # noqa: B018
    ("eig",)  # noqa: B018
    ("eigh",)  # noqa: B018
    ("eigh_tridiagonal",)  # noqa: B018
    ("eigvals",)  # noqa: B018
    ("eigvalsh",)  # noqa: B018
    ("einsum",)  # noqa: B018
    ("expm",)  # noqa: B018
    ("gather_mm",)  # noqa: B018
    ("global_norm",)  # noqa: B018
    ("hadamard_transform",)  # noqa: B018
    ("hessenberg",)  # noqa: B018
    ("householder_product",)  # noqa: B018
    ("inner",)  # noqa: B018
    ("inv",)  # noqa: B018
    ("inv_ex",)  # noqa: B018
    ("logdet",)  # noqa: B018
    ("logm",)  # noqa: B018
    ("lstsq",)  # noqa: B018
    ("lu",)  # noqa: B018
    ("lu_factor",)  # noqa: B018
    ("lu_matrix_inverse",)  # noqa: B018
    ("lu_pivots_to_permutation",)  # noqa: B018
    ("lu_reconstruct",)  # noqa: B018
    ("lu_solve",)  # noqa: B018
    ("matmul",)  # noqa: B018
    ("matrix_exp",)  # noqa: B018
    ("matrix_exponential",)  # noqa: B018
    ("matrix_norm",)  # noqa: B018
    ("matrix_power",)  # noqa: B018
    ("matrix_rank",)  # noqa: B018
    ("matrix_transpose",)  # noqa: B018
    ("matvec",)  # noqa: B018
    ("multi_dot",)  # noqa: B018
    ("norm",)  # noqa: B018
    ("normalize",)  # noqa: B018
    ("outer",)  # noqa: B018
    ("pinv",)  # noqa: B018
    ("power_iteration",)  # noqa: B018
    ("qdwh",)  # noqa: B018
    ("qr",)  # noqa: B018
    ("schur",)  # noqa: B018
    ("segmented_mm",)  # noqa: B018
    ("set_diag",)  # noqa: B018
    ("slogdet",)  # noqa: B018
    ("solve",)  # noqa: B018
    ("solve_ex",)  # noqa: B018
    ("solve_triangular",)  # noqa: B018
    ("sqrtm",)  # noqa: B018
    ("svd",)  # noqa: B018
    ("svdvals",)  # noqa: B018
    ("tensor_diag",)  # noqa: B018
    ("tensor_diag_part",)  # noqa: B018
    ("tensordot",)  # noqa: B018
    ("tensorinv",)  # noqa: B018
    ("tensorsolve",)  # noqa: B018
    ("trace",)  # noqa: B018
    ("tri_inv",)  # noqa: B018
    ("triangular_solve",)  # noqa: B018
    ("tridiagonal",)  # noqa: B018
    ("tridiagonal_matmul",)  # noqa: B018
    ("tridiagonal_solve",)  # noqa: B018
    ("vdot",)  # noqa: B018
    ("vecdot",)  # noqa: B018
    ("vector_norm",)  # noqa: B018


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
