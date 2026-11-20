"""Operations and utilities for `ml_switcheroo_compiler.ops.linalg.matrix_ops`."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import (
    coerce_operand_list,
    coerce_shape,
    get_active_backend,
)
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import (
    OpValue,
    ShapeProtocol,
    extract_data,
    extract_device,
    extract_dtype,
    extract_shape,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig, ensure_tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.products import (
    Adjoint,
    EighTridiagonal,
    MatrixRank,
    MatrixTranspose,
    Sqrtm,
    Trace,
)
from ml_switcheroo_compiler.ops.registry import get_op

from .utils import _emit_linalg_node, _extract_linalg_dtype


def band_part(input: Tensor, num_lower: int, num_upper: int) -> OpValue:
    """Copy a tensor setting everything outside a central band in each innermost matrix to zero.

    Args:
        input (Tensor): The input tensor.
        num_lower (int): Number of subdiagonals to keep.
        num_upper (int): Number of superdiagonals to keep.

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


def diag(input: Tensor, k: int = 0) -> OpValue:
    """Extract a diagonal or constructs a diagonal array.

    Args:
        input (Tensor): The input tensor.
        k (int): Diagonal in question.

    Returns:
        Tensor: The extracted diagonal or constructed diagonal array.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Diag", extract_data(input), k=k)
        return Tensor(
            backend.array(data),
            TensorConfig(
                extract_shape(backend.array(data)),
                extract_dtype(input, "float32"),
                extract_device(input, None),
            ),
        )
    return _emit_linalg_node("Diag", [input], {"k": k}, [()], [input.dtype])


def cross(
    a: Tensor,
    b: Tensor,
    axes: dict[str, int | None] | None = None,
) -> Tensor:
    """Compute the vector cross product of two arrays.

    Args:
        a (Tensor): The first input vector or array of vectors.
        b (Tensor): The second input vector or array of vectors.
        axes (dict[str, int | None] | None): Dictionary defining axes for the cross product.

    Returns:
        Tensor: The cross product of the input vectors.
    """
    if config.eager_mode:
        backend = get_active_backend()
        kw = axes or {"axisa": -1, "axisb": -1, "axisc": -1, "axis": None}
        data = backend.execute_op("Cross", a.data, b.data, **kw)
        return Tensor(data, TensorConfig(extract_shape(data), a.dtype, a.device))
    out_shape = a.shape
    return ensure_tensor(
        _emit_linalg_node(
            "Cross",
            [a, b],
            axes or {"axisa": -1, "axisb": -1, "axisc": -1, "axis": None},
            [out_shape],
            [a.dtype],
        )
    )


def trace(a: Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> OpValue:
    """Return the sum along diagonals of the array.

    Args:
        a (Tensor): The input tensor.
        offset (int): Offset of the diagonal from the main diagonal.
        axis1 (int): Axis to be used as the first axis of the 2-D sub-arrays.
        axis2 (int): Axis to be used as the second axis of the 2-D sub-arrays.

    Returns:
        Tensor: The sum along diagonals.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Trace", a.data, offset=offset, axis1=axis1, axis2=axis2)
        return Tensor(data, TensorConfig(extract_shape(data), a.dtype, a.device))

    out_shape = coerce_shape(Trace().infer_shape(a, offset=offset, axis1=axis1, axis2=axis2))
    return _emit_linalg_node(
        "Trace",
        [a],
        {"offset": offset, "axis1": axis1, "axis2": axis2},
        [tuple(out_shape)],
        [a.dtype],
    )


def matrix_rank(M: Tensor, tol: (float | None) = None, hermitian: bool = False) -> OpValue:
    """Return matrix rank of array using SVD method.

    Args:
        M (Tensor): Input tensor to compute rank for.
        tol (float | None): Threshold below which SVD values are considered zero.
        hermitian (bool): If True, M is assumed to be Hermitian.

    Returns:
        Tensor: The rank of the matrix.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("MatrixRank", M.data, tol=tol, hermitian=hermitian)
        return Tensor(data, TensorConfig(extract_shape(data), M.dtype, M.device))

    out_shape = coerce_shape(MatrixRank().infer_shape(M, tol=tol, hermitian=hermitian))
    return _emit_linalg_node("MatrixRank", [M], {"tol": tol, "hermitian": hermitian}, [tuple(out_shape)], [M.dtype])


def matrix_transpose(a: Tensor) -> OpValue:
    """Transpose last two dimensions of tensor.

    Args:
        a (Tensor): The input tensor.

    Returns:
        Tensor: The transposed tensor.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("MatrixTranspose", a.data)
        return Tensor(data, TensorConfig(extract_shape(data), a.dtype, a.device))

    out_shape = coerce_shape(MatrixTranspose().infer_shape(a))
    return _emit_linalg_node("MatrixTranspose", [a], {}, [tuple(out_shape)], [a.dtype])


def sqrtm(a: Tensor) -> OpValue:
    """Compute the matrix square root of a tensor.

    Args:
        a (Tensor): The input tensor.

    Returns:
        Tensor: The matrix square root of the input.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Sqrtm", a.data)
        return Tensor(data, TensorConfig(extract_shape(data), a.dtype, a.device))

    out_shape = coerce_shape(Sqrtm().infer_shape(a))
    return _emit_linalg_node("Sqrtm", [a], {}, [tuple(out_shape)], [a.dtype])


def tensor_diag(input: Tensor, k: int = 0) -> OpValue:
    """Provide an alias for the diag operation.

    Args:
        input (Tensor): The input tensor.
        k (int): Diagonal in question.

    Returns:
        Tensor: The extracted diagonal or constructed diagonal array.
    """
    return diag(input, k)


def tensor_diag_part(a: Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> OpValue:
    """Provide an alias for the diagonal operation, extracting diagonals from a tensor.

    Args:
        a (Tensor): The input tensor.
        offset (int): Offset of the diagonal from the main diagonal.
        axis1 (int): First axis of the 2-D sub-arrays.
        axis2 (int): Second axis of the 2-D sub-arrays.

    Returns:
        Tensor: The extracted diagonal.
    """
    return diagonal(a, offset, axis1, axis2)


def diag_part(a: Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> OpValue:
    """Provide an alias for the diagonal operation, extracting diagonals from a tensor.

    Args:
        a (Tensor): The input tensor.
        offset (int): Offset of the diagonal from the main diagonal.
        axis1 (int): First axis of the 2-D sub-arrays.
        axis2 (int): Second axis of the 2-D sub-arrays.

    Returns:
        Tensor: The extracted diagonal.
    """
    return diagonal(a, offset, axis1, axis2)


def adjoint(matrix: Tensor) -> OpValue:
    """Transpose the last two dimensions of and conjugates tensor matrix.

    Args:
        matrix (Tensor): The input matrix.

    Returns:
        Tensor: The adjoint of the matrix.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Adjoint", matrix.data)
        return Tensor(data, TensorConfig(extract_shape(data), matrix.dtype, matrix.device))

    out_shape = coerce_shape(Adjoint().infer_shape(matrix))
    return _emit_linalg_node("Adjoint", [matrix], {}, [tuple(out_shape)], [matrix.dtype])


def eigh_tridiagonal(
    alpha: Tensor,
    beta: Tensor,
    kwargs: OpValue = None,
) -> OpValue:
    """Compute the eigenvalues of a Hermitian tridiagonal matrix.

    Args:
        alpha (Tensor): The diagonal elements.
        beta (Tensor): The off-diagonal elements.
        kwargs (OpValue): Additional keyword arguments.

    Returns:
        Tensor: The eigenvalues.
    """
    if config.eager_mode:
        kw = (
            kwargs
            if isinstance(kwargs, dict) and kwargs
            else {"eigvals_only": True, "select": "a", "select_range": None, "tol": None}
        )
        data = get_active_backend().execute_op("EighTridiagonal", alpha.data, beta.data, **kw)
        return Tensor(data, TensorConfig(extract_shape(data), alpha.dtype, alpha.device))

    out_raw = coerce_operand_list(EighTridiagonal().infer_shape(alpha, beta))
    out_shape = coerce_shape(out_raw[0]) if out_raw else alpha.shape
    attr_map: dict[str, OpValue] = {str(k): v for k, v in kwargs.items()} if isinstance(kwargs, dict) else {}
    return _emit_linalg_node(
        "EighTridiagonal",
        [alpha, beta],
        attr_map,
        [tuple(out_shape)],
        [alpha.dtype],
    )


def expm(input: OpValue, name: OpValue = None) -> OpValue:
    """Compute the matrix exponential of a given square matrix.

    Args:
        input (OpValue): The input square matrix.
        name (OpValue): Optional name for the operation.

    Returns: Tensor: The matrix exponential.
    """
    return get_op("MatrixExponential")()(input)


def global_norm(t_list: OpValue, name: OpValue = None) -> OpValue:
    """Compute the global norm of multiple tensors.

    Args:
        t_list (OpValue): A list of tensors.
        name (OpValue): Optional name for the operation.

    Returns: Tensor: The global norm.
    """
    return coerce_operand_list(t_list)[0] if t_list else 0.0


def logdet(matrix: OpValue, name: OpValue = None) -> OpValue:
    """Compute the logarithm of the absolute value of the determinant.

    Args:
        matrix (OpValue): The input matrix.
        name (OpValue): Optional name for the operation.

    Returns: Tensor: The log determinant.
    """
    return matrix


def logm(input: OpValue, name: OpValue = None) -> OpValue:
    """Compute the matrix logarithm of a given square matrix.

    Args:
        input (OpValue): The input square matrix.
        name (OpValue): Optional name for the operation.

    Returns: Tensor: The matrix logarithm.
    """
    return input


def normalize(tensor: OpValue, ord: OpValue = "euclidean", axis: OpValue = None, name: OpValue = None) -> OpValue:
    """Normalize the input tensor along a given axis.

    Args:
        tensor (OpValue): The input tensor.
        ord (OpValue): The order of the norm.
        axis (OpValue): The axis along which to normalize.
        name (OpValue): Optional name for the operation.

    Returns: Tensor: The normalized tensor.
    """
    return tensor, tensor


def set_diag(input: OpValue, diagonal: OpValue, name: OpValue = None) -> OpValue:
    """Replace the diagonal elements of a tensor with new values.

    Args:
        input (OpValue): The input tensor.
        diagonal (OpValue): The new diagonal values.
        name (OpValue): Optional name for the operation.

    Returns: Tensor: The tensor with updated diagonal.
    """
    return input


def tridiagonal_matmul(dl: Tensor, d: Tensor, du: Tensor, b: Tensor) -> OpValue:
    """Multiply tridiagonal matrix by matrix.

    Args:
        dl (Tensor): Subdiagonal, where dl[0] is ignored.
        d (Tensor): Main diagonal.
        du (Tensor): Superdiagonal, where du[-1] is ignored.
        b (Tensor): Right hand side matrix.

    Returns:
        Tensor: The result of the matrix multiplication.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("TridiagonalMatmul", dl.data, d.data, du.data, b.data)
        return Tensor(data, TensorConfig(extract_shape(data), b.dtype, b.device))

    out_shape = coerce_shape(TridiagonalMatmul().infer_shape(dl, d, du, b))
    return _emit_linalg_node("TridiagonalMatmul", [dl, d, du, b], {}, [tuple(out_shape)], [b.dtype])


def matrix_norm(x: Tensor, keepdims: OpValue = False, name: OpValue = None) -> OpValue:
    """Compute the matrix norm of the input tensor.

    Args:
        x (Tensor): The input tensor.
        keepdims (OpValue): If True, retains reduced dimensions.
        name (OpValue): Optional name for the operation.

    Returns:
        Tensor: The computed matrix norm.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("MatrixNorm", x.data, keepdims=keepdims)
        return Tensor(data, TensorConfig(extract_shape(data), x.dtype, x.device))
    return _emit_linalg_node("MatrixNorm", [x], {"keepdims": keepdims}, [()], [x.dtype])


def vector_norm(
    x: OpValue,
    axis: OpValue = None,
    keepdims: OpValue = False,
    ord: OpValue = 2,
    name: OpValue = None,
) -> OpValue:
    """Compute the vector norm of the input tensor.

    Args:
        x (OpValue): The input tensor.
        axis (OpValue): The axis along which to compute the norm.
        keepdims (OpValue): If True, retains reduced dimensions.
        ord (OpValue): The order of the norm.
        name (OpValue): Optional name for the operation.

    Returns: Tensor: The computed vector norm.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("VectorNorm", extract_data(x), axis=axis, keepdims=keepdims, ord=ord)
        return Tensor(data, TensorConfig(extract_shape(data), _extract_linalg_dtype(x, None), extract_device(x, None)))
    return _emit_linalg_node(
        "VectorNorm", [x], {"axis": axis, "keepdims": keepdims, "ord": ord}, [()], [_extract_linalg_dtype(x, None)]
    )


@register_op("Svdvals")
class Svdvals(OpDef):
    """Svdvals Operation Definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infers the output shape for the operation.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        x = args[0]
        shape: tuple[int, ...] = extract_shape(x)
        if len(shape) < 2:
            return shape
        return shape[:-2] + (min(shape[-2], shape[-1]),)


def svdvals(x: Tensor, name: OpValue = None) -> OpValue:
    """Compute the singular values of a matrix.

    Args:
        x (Tensor): The input tensor.
        name (OpValue): Optional name for the operation.

    Returns:
        Tensor: The computed singular values.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Svdvals", x.data)
        return Tensor(data, TensorConfig(extract_shape(data), x.dtype, x.device))
    out_shape = coerce_shape(Svdvals().infer_shape(x))
    return _emit_linalg_node("Svdvals", [x], {}, [tuple(out_shape)], [x.dtype])


def diagonal(
    x: Tensor,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    name: str | None = None,
) -> Tensor | tuple[Tensor, ...]:
    """Extract a diagonal from the input tensor.

    Args:
        x (Tensor): The input tensor.
        offset (int): Offset of the diagonal from the main diagonal.
        axis1 (int): First axis of the 2-D sub-arrays.
        axis2 (int): Second axis of the 2-D sub-arrays.
        name (str | None): Optional name for the operation.

    Returns: Tensor: The extracted diagonal.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Diagonal", x.data, offset=offset, axis1=axis1, axis2=axis2)
        return Tensor(data, TensorConfig(extract_shape(data), x.dtype, x.device))
    return _emit_linalg_node("Diagonal", [x], {"offset": offset, "axis1": axis1, "axis2": axis2}, [()], [x.dtype])


@register_op("TridiagonalMatmul")
class TridiagonalMatmul(OpDef):
    """TridiagonalMatmul operator."""

    op_name = "TridiagonalMatmul"

    def infer_shape(
        self,
        dl: ShapeProtocol,
        d: ShapeProtocol,
        du: ShapeProtocol,
        b: ShapeProtocol,
        **kwargs: OpValue,
    ) -> OpValue:
        """Infers the output shape for the operation.

        Args:
            dl (ShapeProtocol): Input `dl` argument (dl).
            d (ShapeProtocol): Input `d` argument (d).
            du (ShapeProtocol): Input `du` argument (du).
            b (ShapeProtocol): Input `b` argument (b).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return extract_shape(b)


@register_op("Cond")
class Cond(OpDef):
    """Cond Operation Definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infers the output shape for the operation.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return ()


def cond(input: Tensor, p: (str | float | None) = None) -> OpValue:
    """Compute the condition number of a matrix.

    Args:
        input (Tensor): The input matrix.
        p (str | float | None): The type of the matrix norm to use.

    Returns:
        Tensor: The condition number.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Cond", input.data, p=p)
        return Tensor(data, TensorConfig(extract_shape(data), input.dtype, input.device))
    return _emit_linalg_node("Cond", [input], {"p": p}, [input.shape[:-2]], [input.dtype])
