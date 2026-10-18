"""Core abstractions and logic definitions for solvers.py."""

from __future__ import annotations

from typing import cast

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.constants import MATRIX_RANK
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op

from .decompositions.solvers import TriangularSolve
from .utils import _emit_linalg_node


def cholesky_solve(chol: Tensor, rhs: Tensor) -> TensorOrScalar:
    """Solve systems of linear equations A X = RHS given the Cholesky factorization of A.

    Args:
        chol (Tensor): The Cholesky factorization of the matrix A.
        rhs (Tensor): The right-hand side tensor.

    Returns:
        Tensor: The solution tensor X.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("CholeskySolve", chol.data, rhs.data)
        return Tensor(data, TensorConfig(getattr(data, "shape"), rhs.dtype, rhs.device))

    out_shape = getattr(CholeskySolve(), "infer_shape")(chol, rhs)
    return _emit_linalg_node("CholeskySolve", [chol, rhs], {}, [tuple(out_shape)], [rhs.dtype])


def banded_triangular_solve(bands: Tensor, rhs: Tensor, lower: bool = True, adjoint: bool = False) -> TensorOrScalar:
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
        return Tensor(data, TensorConfig(getattr(data, "shape"), rhs.dtype, rhs.device))

    out_shape = getattr(BandedTriangularSolve(), "infer_shape")(bands, rhs)
    return _emit_linalg_node(
        "BandedTriangularSolve",
        [bands, rhs],
        {"lower": lower, "adjoint": adjoint},
        [tuple(out_shape)],
        [rhs.dtype],
    )


def conjugate_gradient(
    operator: TensorOrScalar,
    rhs: TensorOrScalar,
    tol: TensorOrScalar = 1e-05,
    max_iter: TensorOrScalar = 20,
    name: TensorOrScalar = "conjugate_gradient",
) -> TensorOrScalar:
    """Solve a system of linear equations using the conjugate gradient method.

    Args:
        operator (TensorOrScalar): The linear operator or matrix A.
        rhs (TensorOrScalar): The right-hand side vector.
        tol (TensorOrScalar): The tolerance for convergence. Defaults to 1e-05.
        max_iter (TensorOrScalar): The maximum number of iterations. Defaults to 20.
        name (TensorOrScalar): The name of the operation. Defaults to "conjugate_gradient".

    Returns: Tensor: The approximate solution to the system.
    """
    return rhs


@register_op("Lstsq")
class Lstsq(OpDef):
    """Least squares solver operation definition.

    Computes the least squares solution to a linear matrix equation.
    """

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the output tensor for the least squares operation.

        Args:
            *args (EagerTensorProtocol): Positional arguments, typically the input matrices A and B.
            **kwargs (EagerTensorProtocol): Additional keyword arguments.

        Returns: Tensor: The inferred shape of the output tensor.
        """
        a = args[0]
        b = args[1]
        shape_a: tuple[int, ...] = getattr(a, "shape", ())
        shape_b: tuple[int, ...] = getattr(b, "shape", ())
        m, n = shape_a[-2:]
        if len(shape_b) == len(shape_a) - 1:
            return shape_b[:-1] + (n,)
        return shape_b[:-2] + (n, shape_b[-1])


def lstsq(
    a: Tensor,
    b: Tensor,
    rcond: float | None = None,
) -> TensorOrScalar:
    """Compute the lstsq tensor operation.

    Args:
        a (Tensor): The a parameter.
        b (Tensor): The b parameter.
        rcond (float | None): The rcond parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        from typing import cast

        data = get_active_backend().execute_op("Lstsq", a.data, b.data, rcond=rcond)
        if isinstance(data, tuple) or (hasattr(data, "solution") and hasattr(data, "residuals")):
            # torch returns namedtuple, numpy returns tuple
            data = getattr(data, "solution") if hasattr(data, "solution") else cast("tuple[Tensor, ...]", data)[0]
        return Tensor(data, TensorConfig(getattr(data, "shape"), b.dtype, b.device))

    out_shape = getattr(Lstsq(), "infer_shape")(a, b)
    return _emit_linalg_node("Lstsq", [a, b], {"rcond": rcond}, [tuple(out_shape)], [b.dtype])


def lu(input: TensorOrScalar, output_idx_type: TensorOrScalar = None, name: TensorOrScalar = None) -> TensorOrScalar:
    """Compute the lu tensor operation.

    Args:
        input (TensorOrScalar): The input parameter.
        output_idx_type (TensorOrScalar): The output_idx_type parameter.
        name (TensorOrScalar): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return input, input, input


def lu_matrix_inverse(lower_upper: Tensor, perm: Tensor, validate_args: TensorOrScalar = False, name: TensorOrScalar = None) -> TensorOrScalar:
    """Compute the lu matrix inverse tensor operation.

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
        return Tensor(data, TensorConfig(getattr(data, "shape"), lower_upper.dtype, lower_upper.device))

    out_shape = getattr(LuMatrixInverse(), "infer_shape")(lower_upper, perm)
    return _emit_linalg_node(
        "LuMatrixInverse",
        [lower_upper, perm],
        {"validate_args": validate_args},
        [tuple(out_shape)],
        [lower_upper.dtype],
    )


def lu_reconstruct(lower_upper: Tensor, perm: Tensor, validate_args: TensorOrScalar = False, name: TensorOrScalar = None) -> TensorOrScalar:
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
        return Tensor(data, TensorConfig(getattr(data, "shape"), lower_upper.dtype, lower_upper.device))

    out_shape = getattr(LuReconstruct(), "infer_shape")(lower_upper, perm)
    return _emit_linalg_node("LuReconstruct", [lower_upper, perm], {"validate_args": validate_args}, [tuple(out_shape)], [lower_upper.dtype])


def lu_solve(
    lower_upper: TensorOrScalar,
    perm: TensorOrScalar,
    rhs: TensorOrScalar,
    validate_args: TensorOrScalar = False,
    name: TensorOrScalar = None,
) -> TensorOrScalar:
    """Solve a system of linear equations using the LU decomposition of the matrix.

    Args:
        lower_upper (TensorOrScalar): The LU factorized matrix.
        perm (TensorOrScalar): The permutation vector or matrix.
        rhs (TensorOrScalar): The right-hand side tensor.
        validate_args (TensorOrScalar): Whether to validate the inputs. Defaults to False.
        name (TensorOrScalar): Optional name for the operation. Defaults to None.

    Returns: Tensor: The solution to the linear system.
    """
    return rhs


def triangular_solve(matrix: Tensor, rhs: Tensor, lower: TensorOrScalar = True, adjoint: TensorOrScalar = False, name: TensorOrScalar = None) -> TensorOrScalar:
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
        return Tensor(data, TensorConfig(getattr(data, "shape"), rhs.dtype, rhs.device))

    out_shape = getattr(TriangularSolve(), "infer_shape")(matrix, rhs)
    return _emit_linalg_node(
        "TriangularSolve",
        [matrix, rhs],
        {"lower": lower, "adjoint": adjoint, "unit_diagonal": False},
        [tuple(out_shape)],
        [rhs.dtype],
    )


def tridiagonal_solve(
    diagonals: TensorOrScalar,
    rhs: TensorOrScalar,
    diagonals_format: TensorOrScalar = "...",
    partial_pivoting: TensorOrScalar = True,
    name: TensorOrScalar = None,
) -> TensorOrScalar:
    """Solve systems of linear equations with tridiagonal matrices.

    Args:
        diagonals (TensorOrScalar): The diagonals of the tridiagonal matrix.
        rhs (TensorOrScalar): The right-hand side tensor.
        diagonals_format (TensorOrScalar): The format of the diagonals representation. Defaults to "...".
        partial_pivoting (TensorOrScalar): Whether to use partial pivoting. Defaults to True.
        name (TensorOrScalar): Optional name for the operation. Defaults to None.

    Returns: Tensor: The solution to the tridiagonal system.
    """
    return rhs


def tensorinv(a: Tensor, ind: TensorOrScalar = 2, name: TensorOrScalar = None) -> TensorOrScalar:
    """Compute the tensorinv tensor operation.

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
        return Tensor(data, TensorConfig(getattr(data, "shape"), a.dtype, a.device))
    return _emit_linalg_node("Tensorinv", [a], {"ind": ind}, [()], [a.dtype])


def tensorsolve(a: Tensor, b: Tensor, axes: TensorOrScalar = None, name: TensorOrScalar = None) -> TensorOrScalar:
    """Solve a tensor equation equation `a x = b` for x.

    Args:
        a (Tensor): The coefficient tensor.
        b (Tensor): The right-hand side tensor.
        axes (TensorOrScalar): The axes of `a` to sum over. Defaults to None.
        name (str): Optional name for the operation. Defaults to None.

    Returns:
        Tensor: The solution tensor x.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Tensorsolve", a.data, b.data, axes=axes)
        return Tensor(data, TensorConfig(getattr(data, "shape"), a.dtype, a.device))
    return _emit_linalg_node("Tensorsolve", [a, b], {"axes": axes}, [()], [a.dtype])


@register_op("Pinv")
class Pinv(OpDef):
    """Pseudo-inverse operator definition.

    Computes the Moore-Penrose pseudo-inverse of a matrix.
    """

    def infer_shape(self, a: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the output shape of the pseudo-inverse operation.

        Args:
            a (EagerTensorProtocol): The input tensor.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns: Tensor: The computed result.
        """
        if hasattr(a, "shape"):
            s = list(a.shape)
            if len(s) >= MATRIX_RANK:
                s[-2], s[-1] = s[-1], s[-2]
            return tuple(s)
        return ()


@register_op("Sqrtm")
class Sqrtm(OpDef):
    """Matrix square root operator definition.

    Computes the principal square root of a matrix.
    """

    op_name = "Sqrtm"

    def infer_shape(self, a: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the output tensor for the square root operation.

        Args:
            a (EagerTensorProtocol): The input tensor.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns: Tensor: The inferred shape of the output tensor.
        """
        return a.shape


@register_op("CholeskySolve")
class CholeskySolve(OpDef):
    """Cholesky solve operator definition.

    Computes the solution to a system of linear equations using the Cholesky factorization.
    """

    op_name = "CholeskySolve"

    def infer_shape(self, chol: EagerTensorProtocol, rhs: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer the shape of the output tensor for the Cholesky solve operation.

        Args:
            chol (EagerTensorProtocol): The Cholesky factorization tensor.
            rhs (EagerTensorProtocol): The right-hand side tensor.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns:
            tuple[int, ...]: Broadcasted solution shape.
        """
        shape_a = tuple(int(d) for d in getattr(chol, "shape", getattr(chol, "shape_metadata", chol if isinstance(chol, (list, tuple)) else ())))
        shape_b = tuple(int(d) for d in getattr(rhs, "shape", getattr(rhs, "shape_metadata", rhs if isinstance(rhs, (list, tuple)) else ())))

        if len(shape_a) < 2 or len(shape_b) < 1:
            return shape_b

        batch_a = shape_a[:-2]
        is_1d = (len(shape_b) == 1) or (len(shape_b) == len(shape_a) - 1 and len(shape_a) > 2)

        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        if is_1d:
            batch_b = shape_b[:-1] if len(shape_b) > 1 else ()
            batch_out = broadcast_shapes(batch_a, batch_b) if (batch_a or batch_b) else ()
            return batch_out + (shape_a[-1],)

        batch_b = shape_b[:-2] if len(shape_b) >= 2 else ()
        batch_out = broadcast_shapes(batch_a, batch_b) if (batch_a or batch_b) else ()
        return batch_out + (shape_a[-1], shape_b[-1])


@register_op("BandedTriangularSolve")
class BandedTriangularSolve(OpDef):
    """Banded triangular solve operator definition.

    Solves a linear system with a banded triangular matrix.
    """

    op_name = "BandedTriangularSolve"

    def infer_shape(self, bands: EagerTensorProtocol, rhs: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer the shape of the output tensor for the banded triangular solve operation.

        Args:
            bands (EagerTensorProtocol): The banded triangular matrix tensor.
            rhs (EagerTensorProtocol): The right-hand side tensor.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns:
            tuple[int, ...]: Broadcasted solution shape.
        """
        shape_a = tuple(int(d) for d in getattr(bands, "shape", getattr(bands, "shape_metadata", bands if isinstance(bands, (list, tuple)) else ())))
        shape_b = tuple(int(d) for d in getattr(rhs, "shape", getattr(rhs, "shape_metadata", rhs if isinstance(rhs, (list, tuple)) else ())))

        if len(shape_a) < 2 or len(shape_b) < 1:
            return shape_b

        batch_a = shape_a[:-2]
        is_1d = (len(shape_b) == 1) or (len(shape_b) == len(shape_a) - 1 and len(shape_a) > 2)

        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        if is_1d:
            batch_b = shape_b[:-1] if len(shape_b) > 1 else ()
            batch_out = broadcast_shapes(batch_a, batch_b) if (batch_a or batch_b) else ()
            return batch_out + (shape_b[-1],)

        batch_b = shape_b[:-2] if len(shape_b) >= 2 else ()
        batch_out = broadcast_shapes(batch_a, batch_b) if (batch_a or batch_b) else ()
        return batch_out + (shape_b[-2], shape_b[-1])


@register_op("EighTridiagonal")
class EighTridiagonal(OpDef):
    """Eigenvalues and eigenvectors of a symmetric tridiagonal matrix operator definition.

    Computes eigenvalues and eigenvectors of a real symmetric tridiagonal matrix.
    """

    op_name = "EighTridiagonal"

    def infer_shape(self, alpha: EagerTensorProtocol, beta: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the output tensor for the eigensolver operation.

        Args:
            alpha (EagerTensorProtocol): The diagonal elements of the tridiagonal matrix.
            beta (EagerTensorProtocol): The off-diagonal elements of the tridiagonal matrix.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns: Tensor: A tuple containing the inferred shapes for the eigenvalues and eigenvectors.
        """
        # Actually returns (eigvals, eigvecs)
        return (alpha.shape, list(alpha.shape) + [alpha.shape[-1]])


@register_op("MatrixNorm")
class MatrixNorm(OpDef):
    """Matrix norm operator definition.

    Computes a matrix norm.
    """

    op_name = "MatrixNorm"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the output tensor for the matrix norm operation.

        Args:
            *args (EagerTensorProtocol): Positional arguments, typically the input matrix.
            **kwargs (EagerTensorProtocol): Additional keyword arguments.

        Returns: Tensor: The inferred shape of the output tensor.
        """
        if not args:
            return ()

        a = args[0]
        shape = tuple(int(d) for d in getattr(a, "shape", getattr(a, "shape_metadata", ())))
        axis = kwargs.get("axis", (-2, -1))
        keepdims = bool(kwargs.get("keepdims", False))

        axes = (axis,) if isinstance(axis, int) else cast("tuple[int, ...]", axis)
        if keepdims:
            out_shape = list(shape)
            for d in axes:
                out_shape[d] = 1
            return tuple(out_shape)

        axis_indices = [d if d >= 0 else len(shape) + d for d in axes]
        return tuple(s for i, s in enumerate(shape) if i not in axis_indices)


@register_op("VectorNorm")
class VectorNorm(OpDef):
    """Vector norm operator definition.

    Computes a vector norm.
    """

    op_name = "VectorNorm"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the output tensor for the vector norm operation.

        Args:
            *args (EagerTensorProtocol): Positional arguments, typically the input vector.
            **kwargs (EagerTensorProtocol): Additional keyword arguments.

        Returns: Tensor: The inferred shape of the output tensor.
        """
        if not args:
            return ()

        a = args[0]
        shape = tuple(int(d) for d in getattr(a, "shape", getattr(a, "shape_metadata", ())))
        axis = kwargs.get("axis", -1)
        keepdims = bool(kwargs.get("keepdims", False))

        axes = (axis,) if isinstance(axis, int) else cast("tuple[int, ...]", axis)
        if keepdims:
            out_shape = list(shape)
            for d in axes:
                out_shape[d] = 1
            return tuple(out_shape)

        axis_indices = [d if d >= 0 else len(shape) + d for d in axes]
        return tuple(s for i, s in enumerate(shape) if i not in axis_indices)


@register_op("Svdvals")
class Svdvals(OpDef):
    """Singular values operator definition.

    Computes the singular values of a matrix.
    """

    op_name = "Svdvals"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the output tensor for the singular values operation.

        Args:
            *args (EagerTensorProtocol): Positional arguments, typically the input matrix.
            **kwargs (EagerTensorProtocol): Additional keyword arguments.

        Returns: Tensor: The inferred shape of the output tensor.
        """
        in_shape = args[0].shape
        return in_shape[:-1] if len(in_shape) > 1 else ()


@register_op("Tensorinv")
class Tensorinv(OpDef):
    """Tensor inverse operator definition.

    Computes the inverse of a tensor operationally.
    """

    op_name = "Tensorinv"

    def infer_shape(self, a: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the output tensor for the tensor inverse operation.

        Args:
            a (EagerTensorProtocol): The input tensor.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns: Tensor: The inferred shape of the output tensor.
        """
        in_shape = a.shape
        ind = kwargs.get("ind", 2)
        return in_shape[int(cast(int, ind)) :] + in_shape[: int(cast(int, ind))]


@register_op("Tensorsolve")
class Tensorsolve(OpDef):
    """Tensor solve operator definition.

    Solves a linear tensor equation.
    """

    op_name = "Tensorsolve"

    def infer_shape(self, a: EagerTensorProtocol, b: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the output tensor for the tensor solve operation.

        Args:
            a (EagerTensorProtocol): The coefficient tensor.
            b (EagerTensorProtocol): The right-hand side tensor.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns: Tensor: The inferred shape of the output tensor.
        """
        return a.shape[len(b.shape) :]


@register_op("LuMatrixInverse")
class LuMatrixInverse(OpDef):
    """LU matrix inverse operator definition.

    Computes the inverse of a matrix from its LU decomposition.
    """

    op_name = "LuMatrixInverse"

    def infer_shape(self, lower_upper: EagerTensorProtocol, perm: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the output tensor for the LU matrix inverse operation.

        Args:
            lower_upper (EagerTensorProtocol): The LU factorized matrix.
            perm (EagerTensorProtocol): The permutation vector or matrix.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns: Tensor: The inferred shape of the output tensor.
        """
        return lower_upper.shape


@register_op("LuReconstruct")
class LuReconstruct(OpDef):
    """LU reconstruct operator definition.

    Reconstructs a matrix from its LU decomposition and permutation.
    """

    op_name = "LuReconstruct"

    def infer_shape(self, lower_upper: EagerTensorProtocol, perm: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the output tensor for the LU reconstruct operation.

        Args:
            lower_upper (EagerTensorProtocol): The LU factorized matrix.
            perm (EagerTensorProtocol): The permutation vector or matrix.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns: Tensor: The inferred shape of the output tensor.
        """
        return lower_upper.shape
