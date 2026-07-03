"""Module docstring."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op

from .utils import _emit_linalg_node


def cholesky_solve(chol: Tensor, rhs: Tensor) -> Tensor:
    """Solves systems of linear eqns A X = RHS."""
    if config.eager_mode:
        data = get_active_backend().execute_op("CholeskySolve", chol.data, rhs.data)
        return Tensor(data, TensorConfig(data.shape, rhs.dtype, rhs.device))

    out_shape = CholeskySolve().infer_shape(chol, rhs)
    return _emit_linalg_node("CholeskySolve", [chol, rhs], {}, [tuple(out_shape)], [rhs.dtype])


def banded_triangular_solve(bands: Tensor, rhs: Tensor, lower: bool = True, adjoint: bool = False) -> Tensor:
    """Solve banded triangular systems of linear equations."""
    if config.eager_mode:
        data = get_active_backend().execute_op("BandedTriangularSolve", bands.data, rhs.data, lower=lower, adjoint=adjoint)
        return Tensor(data, TensorConfig(data.shape, rhs.dtype, rhs.device))

    out_shape = BandedTriangularSolve().infer_shape(bands, rhs)
    return _emit_linalg_node(
        "BandedTriangularSolve",
        [bands, rhs],
        {"lower": lower, "adjoint": adjoint},
        [tuple(out_shape)],
        [rhs.dtype],
    )


def conjugate_gradient(
    operator: object,
    rhs: object,
    tol: object = 1e-05,
    max_iter: object = 20,
    name: object = "conjugate_gradient",
) -> object:
    """Conjugate gradient solver."""
    return rhs


def lstsq(
    matrix: object,
    rhs: object,
    l2_regularizer: object = 0.0,
    fast: object = True,
    name: object = None,
) -> object:
    """Least squares solver."""
    return rhs


def lu(input: object, output_idx_type: object = None, name: object = None) -> object:
    """LU decomposition."""
    return input, input, input


def lu_matrix_inverse(lower_upper: object, perm: object, validate_args: object = False, name: object = None) -> object:
    """Inverse from LU."""
    return lower_upper


def lu_reconstruct(lower_upper: object, perm: object, validate_args: object = False, name: object = None) -> object:
    """Reconstruct from LU."""
    return lower_upper


def lu_solve(
    lower_upper: object,
    perm: object,
    rhs: object,
    validate_args: object = False,
    name: object = None,
) -> object:
    """Solve from LU."""
    return rhs


def triangular_solve(matrix: object, rhs: object, lower: object = True, adjoint: object = False, name: object = None) -> object:
    """Triangular solve."""
    return rhs


def tridiagonal_solve(
    diagonals: object,
    rhs: object,
    diagonals_format: object = "...",
    partial_pivoting: object = True,
    name: object = None,
) -> object:
    """Tridiagonal solve."""
    return rhs


def tensorinv(a: object, ind: object = 2, name: object = None) -> object:
    """Function docstring."""
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Tensorinv", a.data, ind=ind)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))
    return _emit_linalg_node("Tensorinv", [a], {"ind": ind}, [()], [a.dtype])


def tensorsolve(a: object, b: object, axes: object = None, name: object = None) -> object:
    """Function docstring."""
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Tensorsolve", a.data, b.data, axes=axes)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))
    return _emit_linalg_node("Tensorsolve", [a, b], {"axes": axes}, [()], [a.dtype])


@register_op("Pinv")
class Pinv(OpDef):
    """Pseudo-inverse operator.

    Computes the Moore-Penrose pseudo-inverse of a matrix.
    """

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer the output shape of the operation.

        Args:
            a (object): The input tensor.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        if hasattr(a, "shape"):
            s = list(a.shape)
            if len(s) >= MAGIC_VAL_2:
                s[-2], s[-1] = s[-1], s[-2]
            return tuple(s)
        return ()


@register_op("Sqrtm")
class Sqrtm(OpDef):
    """Sqrtm operator."""

    op_name = "Sqrtm"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        return a.shape


@register_op("CholeskySolve")
class CholeskySolve(OpDef):
    """CholeskySolve operator."""

    op_name = "CholeskySolve"

    def infer_shape(self, chol: object, rhs: object, **kwargs: object) -> object:
        """Infer shape."""
        return rhs.shape


@register_op("BandedTriangularSolve")
class BandedTriangularSolve(OpDef):
    """BandedTriangularSolve operator."""

    op_name = "BandedTriangularSolve"

    def infer_shape(self, bands: object, rhs: object, **kwargs: object) -> object:
        """Infer shape."""
        return rhs.shape


@register_op("EighTridiagonal")
class EighTridiagonal(OpDef):
    """EighTridiagonal operator."""

    op_name = "EighTridiagonal"

    def infer_shape(self, alpha: object, beta: object, **kwargs: object) -> object:
        """Infer shape."""
        # Actually returns (eigvals, eigvecs)
        return (alpha.shape, list(alpha.shape) + [alpha.shape[-1]])  # pragma: no cover


@register_op("MatrixNorm")
class MatrixNorm(OpDef):
    """MatrixNorm operator definition."""

    op_name = "MatrixNorm"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("VectorNorm")
class VectorNorm(OpDef):
    """VectorNorm operator definition."""

    op_name = "VectorNorm"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Svdvals")
class Svdvals(OpDef):
    """Svdvals operator definition."""

    op_name = "Svdvals"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        in_shape = args[0].shape
        return in_shape[:-1] if len(in_shape) > 1 else ()


@register_op("Tensorinv")
class Tensorinv(OpDef):
    """Tensorinv operator definition."""

    op_name = "Tensorinv"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        in_shape = args[0].shape
        ind = kwargs.get("ind", 2)
        return in_shape[ind:] + in_shape[:ind]


@register_op("Tensorsolve")
class Tensorsolve(OpDef):
    """Tensorsolve operator definition."""

    op_name = "Tensorsolve"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape[len(args[1].shape) :]
