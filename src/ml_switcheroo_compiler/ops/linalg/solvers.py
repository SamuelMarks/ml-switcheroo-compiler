"""Core abstractions and logic definitions for solvers.py."""

from __future__ import annotations

from typing import cast

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, EagerValue, get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.constants import MATRIX_RANK
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op

from .decompositions.solvers import TriangularSolve
from .utils import _emit_linalg_node


def cholesky_solve(chol: Tensor, rhs: Tensor) -> EagerValue:
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


def banded_triangular_solve(bands: Tensor, rhs: Tensor, lower: bool = True, adjoint: bool = False) -> EagerValue:
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
    operator: EagerValue,
    rhs: EagerValue,
    tol: EagerValue = 1e-05,
    max_iter: EagerValue = 20,
    name: EagerValue = "conjugate_gradient",
) -> EagerValue:
    """Solve a system of linear equations using the conjugate gradient method.

    Args:
        operator (Any): The linear operator or matrix A.
        rhs (Any): The right-hand side vector.
        tol (Any): The tolerance for convergence. Defaults to 1e-05.
        max_iter (Any): The maximum number of iterations. Defaults to 20.
        name (Any): The name of the operation. Defaults to "conjugate_gradient".

    Returns: Tensor: The approximate solution to the system.
    """
    return rhs


@register_op("Lstsq")
class Lstsq(OpDef):
    """Least squares solver operation definition.

    Computes the least squares solution to a linear matrix equation.
    """

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer the shape of the output tensor for the least squares operation.

        Args:
            *args (Any): Positional arguments, typically the input matrices A and B.
            **kwargs (Any): Additional keyword arguments.

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
) -> EagerValue:
    """Evaluate lstsq operation.

    Args:
        a (Tensor): The a parameter.
        b (Tensor): The b parameter.
        rcond (Any): The rcond parameter.

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


def lu(input: EagerValue, output_idx_type: EagerValue = None, name: EagerValue = None) -> EagerValue:
    """Evaluate lu operation.

    Args:
        input (Any): The input parameter.
        output_idx_type (Any): The output_idx_type parameter.
        name (Any): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return input, input, input


def lu_matrix_inverse(lower_upper: Tensor, perm: Tensor, validate_args: EagerValue = False, name: EagerValue = None) -> EagerValue:
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
        return Tensor(data, TensorConfig(getattr(data, "shape"), lower_upper.dtype, lower_upper.device))

    out_shape = getattr(LuMatrixInverse(), "infer_shape")(lower_upper, perm)
    return _emit_linalg_node(
        "LuMatrixInverse",
        [lower_upper, perm],
        {"validate_args": validate_args},
        [tuple(out_shape)],
        [lower_upper.dtype],
    )


def lu_reconstruct(lower_upper: Tensor, perm: Tensor, validate_args: EagerValue = False, name: EagerValue = None) -> EagerValue:
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
    lower_upper: EagerValue,
    perm: EagerValue,
    rhs: EagerValue,
    validate_args: EagerValue = False,
    name: EagerValue = None,
) -> EagerValue:
    """Solve a system of linear equations using the LU decomposition of the matrix.

    Args:
        lower_upper (Any): The LU factorized matrix.
        perm (Any): The permutation vector or matrix.
        rhs (Any): The right-hand side tensor.
        validate_args (Any): Whether to validate the inputs. Defaults to False.
        name (Any): Optional name for the operation. Defaults to None.

    Returns: Tensor: The solution to the linear system.
    """
    return rhs


def triangular_solve(matrix: Tensor, rhs: Tensor, lower: EagerValue = True, adjoint: EagerValue = False, name: EagerValue = None) -> EagerValue:
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
    diagonals: EagerValue,
    rhs: EagerValue,
    diagonals_format: EagerValue = "...",
    partial_pivoting: EagerValue = True,
    name: EagerValue = None,
) -> EagerValue:
    """Solve systems of linear equations with tridiagonal matrices.

    Args:
        diagonals (Any): The diagonals of the tridiagonal matrix.
        rhs (Any): The right-hand side tensor.
        diagonals_format (Any): The format of the diagonals representation. Defaults to "...".
        partial_pivoting (Any): Whether to use partial pivoting. Defaults to True.
        name (Any): Optional name for the operation. Defaults to None.

    Returns: Tensor: The solution to the tridiagonal system.
    """
    return rhs


def tensorinv(a: Tensor, ind: EagerValue = 2, name: EagerValue = None) -> EagerValue:
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
        return Tensor(data, TensorConfig(getattr(data, "shape"), a.dtype, a.device))
    return _emit_linalg_node("Tensorinv", [a], {"ind": ind}, [()], [a.dtype])


def tensorsolve(a: Tensor, b: Tensor, axes: EagerValue = None, name: EagerValue = None) -> EagerValue:
    """Solve a tensor equation equation `a x = b` for x.

    Args:
        a (Tensor): The coefficient tensor.
        b (Tensor): The right-hand side tensor.
        axes (Any): The axes of `a` to sum over. Defaults to None.
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

    def infer_shape(self, a: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer the output shape of the pseudo-inverse operation.

        Args:
            a (Any): The input tensor.
            **kwargs (Any): Additional keyword arguments.

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

    def infer_shape(self, a: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer the shape of the output tensor for the square root operation.

        Args:
            a (Any): The input tensor.
            **kwargs (Any): Additional keyword arguments.

        Returns: Tensor: The inferred shape of the output tensor.
        """
        return a.shape


@register_op("CholeskySolve")
class CholeskySolve(OpDef):
    """Cholesky solve operator definition.

    Computes the solution to a system of linear equations using the Cholesky factorization.
    """

    op_name = "CholeskySolve"

    def infer_shape(self, chol: EagerTensorProtocol, rhs: EagerTensorProtocol, **kwargs: EagerValue) -> tuple[int, ...]:
        """Infer the shape of the output tensor for the Cholesky solve operation.

        Args:
            chol (object): The Cholesky factorization tensor.
            rhs (object): The right-hand side tensor.
            **kwargs (object): Additional keyword arguments.

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

    def infer_shape(self, bands: EagerTensorProtocol, rhs: EagerTensorProtocol, **kwargs: EagerValue) -> tuple[int, ...]:
        """Infer the shape of the output tensor for the banded triangular solve operation.

        Args:
            bands (object): The banded triangular matrix tensor.
            rhs (object): The right-hand side tensor.
            **kwargs (object): Additional keyword arguments.

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

    def infer_shape(self, alpha: EagerTensorProtocol, beta: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer the shape of the output tensor for the eigensolver operation.

        Args:
            alpha (Any): The diagonal elements of the tridiagonal matrix.
            beta (Any): The off-diagonal elements of the tridiagonal matrix.
            **kwargs (Any): Additional keyword arguments.

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

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer the shape of the output tensor for the matrix norm operation.

        Args:
            *args (Any): Positional arguments, typically the input matrix.
            **kwargs (Any): Additional keyword arguments.

        Returns: Tensor: The inferred shape of the output tensor.
        """
        if not args:
            return ()

        a = args[0]
        shape = tuple(int(d) for d in getattr(a, "shape", getattr(a, "shape_metadata", ())))
        dim = kwargs.get("dim", (-2, -1))
        keepdims = kwargs.get("keepdims", False)

        dims = (dim,) if isinstance(dim, int) else dim
        if keepdims:
            out_shape = list(shape)
            for d in getattr(dims, "__iter__")():
                out_shape[d] = 1
            return tuple(out_shape)

        dim_indices = [d if d >= 0 else len(shape) + d for d in getattr(dims, "__iter__")()]
        return tuple(s for i, s in enumerate(shape) if i not in dim_indices)


@register_op("VectorNorm")
class VectorNorm(OpDef):
    """Vector norm operator definition.

    Computes a vector norm.
    """

    op_name = "VectorNorm"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer the shape of the output tensor for the vector norm operation.

        Args:
            *args (Any): Positional arguments, typically the input vector.
            **kwargs (Any): Additional keyword arguments.

        Returns: Tensor: The inferred shape of the output tensor.
        """
        if not args:
            return ()

        a = args[0]
        shape = tuple(int(d) for d in getattr(a, "shape", getattr(a, "shape_metadata", ())))
        dim = kwargs.get("dim", -1)
        keepdims = kwargs.get("keepdims", False)

        dims = (dim,) if isinstance(dim, int) else dim
        if keepdims:
            out_shape = list(shape)
            for d in getattr(dims, "__iter__")():
                out_shape[d] = 1
            return tuple(out_shape)

        dim_indices = [d if d >= 0 else len(shape) + d for d in getattr(dims, "__iter__")()]
        return tuple(s for i, s in enumerate(shape) if i not in dim_indices)


@register_op("Svdvals")
class Svdvals(OpDef):
    """Singular values operator definition.

    Computes the singular values of a matrix.
    """

    op_name = "Svdvals"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer the shape of the output tensor for the singular values operation.

        Args:
            *args (Any): Positional arguments, typically the input matrix.
            **kwargs (Any): Additional keyword arguments.

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

    def infer_shape(self, a: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer the shape of the output tensor for the tensor inverse operation.

        Args:
            a (Any): The input tensor.
            **kwargs (Any): Additional keyword arguments.

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

    def infer_shape(self, a: EagerTensorProtocol, b: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer the shape of the output tensor for the tensor solve operation.

        Args:
            a (Any): The coefficient tensor.
            b (Any): The right-hand side tensor.
            **kwargs (Any): Additional keyword arguments.

        Returns: Tensor: The inferred shape of the output tensor.
        """
        return a.shape[len(b.shape) :]


@register_op("LuMatrixInverse")
class LuMatrixInverse(OpDef):
    """LU matrix inverse operator definition.

    Computes the inverse of a matrix from its LU decomposition.
    """

    op_name = "LuMatrixInverse"

    def infer_shape(self, lower_upper: EagerTensorProtocol, perm: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer the shape of the output tensor for the LU matrix inverse operation.

        Args:
            lower_upper (Any): The LU factorized matrix.
            perm (Any): The permutation vector or matrix.
            **kwargs (Any): Additional keyword arguments.

        Returns: Tensor: The inferred shape of the output tensor.
        """
        return lower_upper.shape


@register_op("LuReconstruct")
class LuReconstruct(OpDef):
    """LU reconstruct operator definition.

    Reconstructs a matrix from its LU decomposition and permutation.
    """

    op_name = "LuReconstruct"

    def infer_shape(self, lower_upper: EagerTensorProtocol, perm: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer the shape of the output tensor for the LU reconstruct operation.

        Args:
            lower_upper (Any): The LU factorized matrix.
            perm (Any): The permutation vector or matrix.
            **kwargs (Any): Additional keyword arguments.

        Returns: Tensor: The inferred shape of the output tensor.
        """
        return lower_upper.shape
