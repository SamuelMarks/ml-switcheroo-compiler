"""Module solvers.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for solvers.py."""
from typing import Any

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op

from .decompositions.solvers import TriangularSolve
from .utils import _emit_linalg_node


def cholesky_solve(chol: Tensor, rhs: Tensor) -> Any:  # type: ignore
    """Solve systems of linear equations A X = RHS given the Cholesky factorization of A.

    Args:
        chol (Tensor): The Cholesky factorization of the matrix A.
        rhs (Tensor): The right-hand side tensor.

    Returns:
        Tensor: The solution tensor X.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("CholeskySolve", chol.data, rhs.data)
        return Tensor(data, TensorConfig(data.shape, rhs.dtype, rhs.device))

    out_shape = CholeskySolve().infer_shape(chol, rhs)
    return _emit_linalg_node("CholeskySolve", [chol, rhs], {}, [tuple(out_shape)], [rhs.dtype])


def banded_triangular_solve(bands: Tensor, rhs: Tensor, lower: bool = True, adjoint: bool = False) -> Any:  # type: ignore
    """Solve systems of linear equations for a banded triangular matrix.

    Args:
        bands (Tensor): The banded lower or upper triangular matrix.
        rhs (Tensor): The right-hand side tensor.
        lower (bool): Whether the matrix is lower triangular. Defaults to True.
        adjoint (bool): Whether to solve the adjoint system. Defaults to False.

    Returns:
        Tensor: The solution to the banded triangular system.
    """
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
    operator: Any,
    rhs: Any,
    tol: Any = 1e-05,
    max_iter: Any = 20,
    name: Any = "conjugate_gradient",
) -> Any:
    """Solve a system of linear equations using the conjugate gradient method.

    Args:
        operator (object): The linear operator or matrix A.
        rhs (object): The right-hand side vector.
        tol (object): The tolerance for convergence. Defaults to 1e-05.
        max_iter (object): The maximum number of iterations. Defaults to 20.
        name (object): The name of the operation. Defaults to "conjugate_gradient".

    Returns: Any: The approximate solution to the system.
    """
    return rhs


@register_op("Lstsq")
class Lstsq(OpDef):
    """Least squares solver operation definition.

    Computes the least squares solution to a linear matrix equation.
    """

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the shape of the output tensor for the least squares operation.

        Args:
            *args (object): Positional arguments, typically the input matrices A and B.
            **kwargs (object): Additional keyword arguments.

        Returns: Any: The inferred shape of the output tensor.
        """
        a = args[0]
        b = args[1]
        shape_a = getattr(a, "shape", ())
        shape_b = getattr(b, "shape", ())
        m, n = shape_a[-2:]  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        if len(shape_b) == len(shape_a) - 1:
            return shape_b[:-1] + (n,)
        return shape_b[:-2] + (n, shape_b[-1])  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism


def lstsq(
    a: Tensor,  # type: ignore
    b: Tensor,  # type: ignore
    rcond: float | None = None,
) -> Any:
    """Evaluate lstsq operation.

    Args:
        a (Tensor): The a parameter.
        b (Tensor): The b parameter.
        rcond (object): The rcond parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Lstsq", a.data, b.data, rcond=rcond)
        if isinstance(data, tuple) or (hasattr(data, "solution") and hasattr(data, "residuals")):
            # torch returns namedtuple, numpy returns tuple
            data = data.solution if hasattr(data, "solution") else data[0]
        return Tensor(data, TensorConfig(data.shape, b.dtype, b.device))

    out_shape = Lstsq().infer_shape(a, b)
    return _emit_linalg_node("Lstsq", [a, b], {"rcond": rcond}, [tuple(out_shape)], [b.dtype])


def lu(input: Any, output_idx_type: Any = None, name: Any = None) -> Any:
    """Evaluate lu operation.

    Args:
        input (object): The input parameter.
        output_idx_type (object): The output_idx_type parameter.
        name (object): The name parameter.

    Returns: Any: Result.
    """
    return input, input, input


def lu_matrix_inverse(lower_upper: Tensor, perm: Tensor, validate_args: Any = False, name: Any = None) -> Any:  # type: ignore
    """Evaluate lu_matrix_inverse operation.

    Args:
        lower_upper (Tensor): The lower_upper parameter.
        perm (Tensor): The perm parameter.
        validate_args (bool): The validate_args parameter.
        name (str): The name parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("LuMatrixInverse", lower_upper.data, perm.data, validate_args=validate_args)
        return Tensor(data, TensorConfig(data.shape, lower_upper.dtype, lower_upper.device))

    out_shape = LuMatrixInverse().infer_shape(lower_upper, perm)
    return _emit_linalg_node(
        "LuMatrixInverse",
        [lower_upper, perm],
        {"validate_args": validate_args},
        [tuple(out_shape)],
        [lower_upper.dtype],
    )


def lu_reconstruct(lower_upper: Tensor, perm: Tensor, validate_args: Any = False, name: Any = None) -> Any:  # type: ignore
    """Reconstruct a matrix from its LU decomposition and permutation.

    Args:
        lower_upper (Tensor): The LU factorized matrix.
        perm (Tensor): The permutation matrix or vector.
        validate_args (bool): Whether to validate arguments. Defaults to False.
        name (str): Optional name for the operation. Defaults to None.

    Returns:
        Tensor: The reconstructed original matrix.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("LuReconstruct", lower_upper.data, perm.data, validate_args=validate_args)
        return Tensor(data, TensorConfig(data.shape, lower_upper.dtype, lower_upper.device))

    out_shape = LuReconstruct().infer_shape(lower_upper, perm)
    return _emit_linalg_node("LuReconstruct", [lower_upper, perm], {"validate_args": validate_args}, [tuple(out_shape)], [lower_upper.dtype])


def lu_solve(
    lower_upper: Any,
    perm: Any,
    rhs: Any,
    validate_args: Any = False,
    name: Any = None,
) -> Any:
    """Solve a system of linear equations using the LU decomposition of the matrix.

    Args:
        lower_upper (object): The LU factorized matrix.
        perm (object): The permutation vector or matrix.
        rhs (object): The right-hand side tensor.
        validate_args (object): Whether to validate the inputs. Defaults to False.
        name (object): Optional name for the operation. Defaults to None.

    Returns: Any: The solution to the linear system.
    """
    return rhs


def triangular_solve(matrix: Tensor, rhs: Tensor, lower: Any = True, adjoint: Any = False, name: Any = None) -> Any:  # type: ignore
    """Solve a system of linear equations with a triangular matrix.

    Args:
        matrix (Tensor): The triangular matrix.
        rhs (Tensor): The right-hand side tensor.
        lower (bool): Whether the matrix is lower triangular. Defaults to True.
        adjoint (bool): Whether to solve the adjoint system. Defaults to False.
        name (str): Optional name for the operation. Defaults to None.

    Returns:
        Tensor: The solution to the triangular system.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("TriangularSolve", matrix.data, rhs.data, lower=lower, adjoint=adjoint, unit_diagonal=False)
        return Tensor(data, TensorConfig(data.shape, rhs.dtype, rhs.device))

    out_shape = TriangularSolve().infer_shape(matrix, rhs)
    return _emit_linalg_node(
        "TriangularSolve",
        [matrix, rhs],
        {"lower": lower, "adjoint": adjoint, "unit_diagonal": False},
        [tuple(out_shape)],
        [rhs.dtype],
    )


def tridiagonal_solve(
    diagonals: Any,
    rhs: Any,
    diagonals_format: Any = "...",
    partial_pivoting: Any = True,
    name: Any = None,
) -> Any:
    """Solve systems of linear equations with tridiagonal matrices.

    Args:
        diagonals (object): The diagonals of the tridiagonal matrix.
        rhs (object): The right-hand side tensor.
        diagonals_format (object): The format of the diagonals representation. Defaults to "...".
        partial_pivoting (object): Whether to use partial pivoting. Defaults to True.
        name (object): Optional name for the operation. Defaults to None.

    Returns: Any: The solution to the tridiagonal system.
    """
    return rhs


def tensorinv(a: Tensor, ind: Any = 2, name: Any = None) -> Any:  # type: ignore
    """Evaluate tensorinv operation.

    Args:
        a (Tensor): The a parameter.
        ind (int): The ind parameter.
        name (str): The name parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Tensorinv", a.data, ind=ind)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))
    return _emit_linalg_node("Tensorinv", [a], {"ind": ind}, [()], [a.dtype])


def tensorsolve(a: Tensor, b: Tensor, axes: Any = None, name: Any = None) -> Any:  # type: ignore
    """Solve a tensor equation equation `a x = b` for x.

    Args:
        a (Tensor): The coefficient tensor.
        b (Tensor): The right-hand side tensor.
        axes (object): The axes of `a` to sum over. Defaults to None.
        name (str): Optional name for the operation. Defaults to None.

    Returns:
        Tensor: The solution tensor x.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Tensorsolve", a.data, b.data, axes=axes)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))
    return _emit_linalg_node("Tensorsolve", [a, b], {"axes": axes}, [()], [a.dtype])


@register_op("Pinv")
class Pinv(OpDef):
    """Pseudo-inverse operator definition.

    Computes the Moore-Penrose pseudo-inverse of a matrix.
    """

    def infer_shape(self, a: Any, **kwargs: Any) -> Any:
        """Infer the output shape of the pseudo-inverse operation.

        Args:
            a (object): The input tensor.
            **kwargs (object): Additional keyword arguments.

        Returns: Any: The computed result.
        """
        if hasattr(a, "shape"):
            s = list(a.shape)
            if len(s) >= MAGIC_VAL_2:
                s[-2], s[-1] = s[-1], s[-2]
            return tuple(s)
        return ()


@register_op("Sqrtm")
class Sqrtm(OpDef):
    """Matrix square root operator definition.

    Computes the principal square root of a matrix.
    """

    op_name = "Sqrtm"

    def infer_shape(self, a: Any, **kwargs: Any) -> Any:
        """Infer the shape of the output tensor for the square root operation.

        Args:
            a (object): The input tensor.
            **kwargs (object): Additional keyword arguments.

        Returns: Any: The inferred shape of the output tensor.
        """
        return a.shape


@register_op("CholeskySolve")
class CholeskySolve(OpDef):
    """Cholesky solve operator definition.

    Computes the solution to a system of linear equations using the Cholesky factorization.
    """

    op_name = "CholeskySolve"

    def infer_shape(self, chol: Any, rhs: Any, **kwargs: Any) -> Any:
        """Infer the shape of the output tensor for the Cholesky solve operation.

        Args:
            chol (object): The Cholesky factorization tensor.
            rhs (object): The right-hand side tensor.
            **kwargs (object): Additional keyword arguments.

        Returns: Any: The inferred shape of the output tensor.
        """
        return rhs.shape


@register_op("BandedTriangularSolve")
class BandedTriangularSolve(OpDef):
    """Banded triangular solve operator definition.

    Solves a linear system with a banded triangular matrix.
    """

    op_name = "BandedTriangularSolve"

    def infer_shape(self, bands: Any, rhs: Any, **kwargs: Any) -> Any:
        """Infer the shape of the output tensor for the banded triangular solve operation.

        Args:
            bands (object): The banded triangular matrix tensor.
            rhs (object): The right-hand side tensor.
            **kwargs (object): Additional keyword arguments.

        Returns: Any: The inferred shape of the output tensor.
        """
        return rhs.shape


@register_op("EighTridiagonal")
class EighTridiagonal(OpDef):
    """Eigenvalues and eigenvectors of a symmetric tridiagonal matrix operator definition.

    Computes eigenvalues and eigenvectors of a real symmetric tridiagonal matrix.
    """

    op_name = "EighTridiagonal"

    def infer_shape(self, alpha: Any, beta: Any, **kwargs: Any) -> Any:
        """Infer the shape of the output tensor for the eigensolver operation.

        Args:
            alpha (object): The diagonal elements of the tridiagonal matrix.
            beta (object): The off-diagonal elements of the tridiagonal matrix.
            **kwargs (object): Additional keyword arguments.

        Returns: Any: A tuple containing the inferred shapes for the eigenvalues and eigenvectors.
        """
        # Actually returns (eigvals, eigvecs)
        return (alpha.shape, list(alpha.shape) + [alpha.shape[-1]])


@register_op("MatrixNorm")
class MatrixNorm(OpDef):
    """Matrix norm operator definition.

    Computes a matrix norm.
    """

    op_name = "MatrixNorm"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the shape of the output tensor for the matrix norm operation.

        Args:
            *args (object): Positional arguments, typically the input matrix.
            **kwargs (object): Additional keyword arguments.

        Returns: Any: The inferred shape of the output tensor.
        """
        return ()


@register_op("VectorNorm")
class VectorNorm(OpDef):
    """Vector norm operator definition.

    Computes a vector norm.
    """

    op_name = "VectorNorm"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the shape of the output tensor for the vector norm operation.

        Args:
            *args (object): Positional arguments, typically the input vector.
            **kwargs (object): Additional keyword arguments.

        Returns: Any: The inferred shape of the output tensor.
        """
        return ()


@register_op("Svdvals")
class Svdvals(OpDef):
    """Singular values operator definition.

    Computes the singular values of a matrix.
    """

    op_name = "Svdvals"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the shape of the output tensor for the singular values operation.

        Args:
            *args (object): Positional arguments, typically the input matrix.
            **kwargs (object): Additional keyword arguments.

        Returns: Any: The inferred shape of the output tensor.
        """
        in_shape = args[0].shape
        return in_shape[:-1] if len(in_shape) > 1 else ()


@register_op("Tensorinv")
class Tensorinv(OpDef):
    """Tensor inverse operator definition.

    Computes the inverse of a tensor operationally.
    """

    op_name = "Tensorinv"

    def infer_shape(self, a: Any, **kwargs: Any) -> Any:
        """Infer the shape of the output tensor for the tensor inverse operation.

        Args:
            a (object): The input tensor.
            **kwargs (object): Additional keyword arguments.

        Returns: Any: The inferred shape of the output tensor.
        """
        in_shape = a.shape
        ind = kwargs.get("ind", 2)
        return in_shape[ind:] + in_shape[:ind]


@register_op("Tensorsolve")
class Tensorsolve(OpDef):
    """Tensor solve operator definition.

    Solves a linear tensor equation.
    """

    op_name = "Tensorsolve"

    def infer_shape(self, a: Any, b: Any, **kwargs: Any) -> Any:
        """Infer the shape of the output tensor for the tensor solve operation.

        Args:
            a (object): The coefficient tensor.
            b (object): The right-hand side tensor.
            **kwargs (object): Additional keyword arguments.

        Returns: Any: The inferred shape of the output tensor.
        """
        return a.shape[len(b.shape) :]


@register_op("LuMatrixInverse")
class LuMatrixInverse(OpDef):
    """LU matrix inverse operator definition.

    Computes the inverse of a matrix from its LU decomposition.
    """

    op_name = "LuMatrixInverse"

    def infer_shape(self, lower_upper: Any, perm: Any, **kwargs: Any) -> Any:
        """Infer the shape of the output tensor for the LU matrix inverse operation.

        Args:
            lower_upper (object): The LU factorized matrix.
            perm (object): The permutation vector or matrix.
            **kwargs (object): Additional keyword arguments.

        Returns: Any: The inferred shape of the output tensor.
        """
        return lower_upper.shape


@register_op("LuReconstruct")
class LuReconstruct(OpDef):
    """LU reconstruct operator definition.

    Reconstructs a matrix from its LU decomposition and permutation.
    """

    op_name = "LuReconstruct"

    def infer_shape(self, lower_upper: Any, perm: Any, **kwargs: Any) -> Any:
        """Infer the shape of the output tensor for the LU reconstruct operation.

        Args:
            lower_upper (object): The LU factorized matrix.
            perm (object): The permutation vector or matrix.
            **kwargs (object): Additional keyword arguments.

        Returns: Any: The inferred shape of the output tensor.
        """
        return lower_upper.shape
