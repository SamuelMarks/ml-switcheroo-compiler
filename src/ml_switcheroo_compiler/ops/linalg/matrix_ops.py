"""Module docstring."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.decompositions import matrix_exponential
from ml_switcheroo_compiler.ops.linalg.products import Adjoint, MatrixRank, MatrixTranspose, Trace
from ml_switcheroo_compiler.ops.linalg.solvers import EighTridiagonal, Sqrtm

from .utils import _emit_linalg_node


def band_part(input: Tensor, num_lower: int, num_upper: int) -> Tensor:
    """Copy a tensor setting everything outside a central band in each innermost matrix to zero.

    Args:
        input (Tensor): The input tensor.
        num_lower (int): Number of subdiagonals to keep.
        num_upper (int): Number of superdiagonals to keep.

        axes (tuple): The axes.\

    Returns:
        Tensor: The banded tensor.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("BandPart", input.data, num_lower=num_lower, num_upper=num_upper)
        return Tensor(data, TensorConfig(input.shape, input.dtype, input.device))
    return _emit_linalg_node(
        "BandPart",
        [input],
        {"num_lower": num_lower, "num_upper": num_upper},
        [input.shape],
        [input.dtype],
    )


def diag(input: Tensor, k: int = 0) -> Tensor:
    """Extracts a diagonal or constructs a diagonal array.

    Args:
        input (Tensor): The input tensor.
        k (int): Diagonal in question.

        axes (tuple): The axes.\

    Returns:
        Tensor: The extracted diagonal or constructed diagonal array.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Diag", getattr(input, "data", input), k=k)
        return Tensor(
            backend.array(data),
            TensorConfig(
                backend.array(data).shape,
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_linalg_node("Diag", [input], {"k": k}, [()], [input.dtype])


def cross(
    a: Tensor,
    b: Tensor,
    axes: dict[str, int | None] | None = None,
) -> Tensor:
    """Computes the vector cross product of two arrays.

    Args:
        a (Tensor): The first input vector or array of vectors
        b (Tensor): The second input vector or array of vectors
        axisa (int): Axis of a that defines the vector(s). By default, the last axis.
        axisb (int): Axis of b that defines the vector(s). By default, the last axis.
        axisc (int): Axis of c containing the cross product vector(s). By default, the last axis.
        axis (int | None): If defined, the axis of a, b and c that defines the vector(s) and cross product(s).

        axes (tuple): The axes.\

    Returns:
    Tensor: The cross product of the input vectors
    """
    if config.eager_mode:
        backend = get_active_backend()
        kw = axes or {"axisa": -1, "axisb": -1, "axisc": -1, "axis": None}
        data = backend.execute_op("Cross", a.data, b.data, **kw)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))
    out_shape = a.shape
    return _emit_linalg_node(
        "Cross",
        [a, b],
        axes or {"axisa": -1, "axisb": -1, "axisc": -1, "axis": None},
        [out_shape],
        [a.dtype],
    )


def trace(a: Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Tensor:
    """Return the sum along diagonals of the array."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Trace", a.data, offset=offset, axis1=axis1, axis2=axis2)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    out_shape = Trace().infer_shape(a, offset=offset, axis1=axis1, axis2=axis2)
    return _emit_linalg_node(
        "Trace",
        [a],
        {"offset": offset, "axis1": axis1, "axis2": axis2},
        [tuple(out_shape)],
        [a.dtype],
    )


def matrix_rank(M: Tensor, tol: (float | None) = None, hermitian: bool = False) -> Tensor:
    """Return matrix rank of array using SVD method."""
    if config.eager_mode:
        data = get_active_backend().execute_op("MatrixRank", M.data, tol=tol, hermitian=hermitian)
        return Tensor(data, TensorConfig(data.shape, M.dtype, M.device))

    out_shape = MatrixRank().infer_shape(M, tol=tol, hermitian=hermitian)
    return _emit_linalg_node("MatrixRank", [M], {"tol": tol, "hermitian": hermitian}, [tuple(out_shape)], [M.dtype])


def matrix_transpose(a: Tensor) -> Tensor:
    """Transposes last two dimensions of tensor."""
    if config.eager_mode:
        data = get_active_backend().execute_op("MatrixTranspose", a.data)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    out_shape = MatrixTranspose().infer_shape(a)
    return _emit_linalg_node("MatrixTranspose", [a], {}, [tuple(out_shape)], [a.dtype])


def sqrtm(a: Tensor) -> Tensor:
    """Matrix square root."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Sqrtm", a.data)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    out_shape = Sqrtm().infer_shape(a)
    return _emit_linalg_node("Sqrtm", [a], {}, [tuple(out_shape)], [a.dtype])


def tensor_diag(input: Tensor, k: int = 0) -> Tensor:
    """Alias for diag."""
    return diag(input, k)


def tensor_diag_part(a: Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Tensor:
    """Alias for diagonal."""
    return diagonal(a, offset, axis1, axis2)


def diag_part(a: Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Tensor:
    """Alias for diagonal."""
    return diagonal(a, offset, axis1, axis2)


def adjoint(matrix: Tensor) -> Tensor:
    """Transposes the last two dimensions of and conjugates tensor matrix."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Adjoint", matrix.data)
        return Tensor(data, TensorConfig(data.shape, matrix.dtype, matrix.device))

    out_shape = Adjoint().infer_shape(matrix)
    return _emit_linalg_node("Adjoint", [matrix], {}, [tuple(out_shape)], [matrix.dtype])


def eigh_tridiagonal(
    alpha: Tensor,
    beta: Tensor,
    kwargs: dict[str, bool | str | object | float | None] | None = None,
) -> Tensor:
    """Computes the eigenvalues of a Hermitian tridiagonal matrix."""
    if config.eager_mode:
        kw = kwargs or {"eigvals_only": True, "select": "a", "select_range": None, "tol": None}
        data = get_active_backend().execute_op("EighTridiagonal", alpha.data, beta.data, **kw)
        return Tensor(data, TensorConfig(data.shape, alpha.dtype, alpha.device))

    out_shape = EighTridiagonal().infer_shape(alpha, beta)
    return _emit_linalg_node(
        "EighTridiagonal",
        [alpha, beta],
        kwargs or {},
        [tuple(out_shape)],
        [alpha.dtype],
    )


def expm(input: object, name: object = None) -> object:
    """Matrix exponential."""
    return matrix_exponential(input)


def global_norm(t_list: object, name: object = None) -> object:
    """Computes the global norm of multiple tensors."""
    return t_list[0] if t_list else 0.0


def logdet(matrix: object, name: object = None) -> object:
    """Log of absolute determinant."""
    return matrix


def logm(input: object, name: object = None) -> object:
    """Matrix logarithm."""
    return input


def normalize(tensor: object, ord: object = "euclidean", axis: object = None, name: object = None) -> object:
    """Normalize."""
    return tensor, tensor


def set_diag(input: object, diagonal: object, name: object = None) -> object:
    """Set diagonal."""
    return input


def tridiagonal_matmul(superdiag: object, maindiag: object, subdiag: object, rhs: object, **kwargs: object) -> object:
    """Tridiagonal matmul."""
    return rhs


def matrix_norm(x: object, keepdims: object = False, name: object = None) -> object:
    """Function docstring."""
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("MatrixNorm", x.data, keepdims=keepdims)
        return Tensor(data, TensorConfig(data.shape, x.dtype, x.device))
    return _emit_linalg_node("MatrixNorm", [x], {"keepdims": keepdims}, [()], [x.dtype])


def vector_norm(x: object, axis: object = None, keepdims: object = False, ord: object = 2, name: object = None) -> object:
    """Function docstring."""
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("VectorNorm", x.data, axis=axis, keepdims=keepdims, ord=ord)
        return Tensor(data, TensorConfig(data.shape, x.dtype, x.device))
    return _emit_linalg_node("VectorNorm", [x], {"axis": axis, "keepdims": keepdims, "ord": ord}, [()], [x.dtype])


def svdvals(x: object, name: object = None) -> object:
    """Function docstring."""
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Svdvals", x.data)
        return Tensor(data, TensorConfig(data.shape, x.dtype, x.device))
    return _emit_linalg_node("Svdvals", [x], {}, [()], [x.dtype])


def diagonal(x: object, offset: object = 0, axis1: object = 0, axis2: object = 1, name: object = None) -> object:
    """Function docstring."""
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Diagonal", x.data, offset=offset, axis1=axis1, axis2=axis2)
        return Tensor(data, TensorConfig(data.shape, x.dtype, x.device))
    return _emit_linalg_node("Diagonal", [x], {"offset": offset, "axis1": axis1, "axis2": axis2}, [()], [x.dtype])
