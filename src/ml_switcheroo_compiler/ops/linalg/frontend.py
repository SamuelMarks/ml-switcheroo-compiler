# ruff: noqa: E701, D101, ANN001, ANN201, ANN202, ANN002, ANN003, D103, PLR0913
# ruff: noqa: PLR0913
"""Provides linear algebra operations for the ml_switcheroo_compiler framework.

This module contains standard linear algebra functions such as matrix multiplication,
decompositions (SVD, QR, Cholesky, LU), solvers, and other tensor operations. It
supports both eager execution using NumPy/SciPy and graph tracing by emitting logical
nodes to the intermediate representation (IR) graph
"""

from __future__ import annotations
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2

from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder


import uuid
from typing import TYPE_CHECKING

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing import ProxyTensor, _tracer


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ml_switcheroo_compiler.core.dtype import DType


def _build_linalg_output_tensors(
    out_ids: list[str],
    out_shapes: Sequence[Sequence[int]],
    out_dtypes: Sequence[DType],
    device: object,
) -> list[Tensor]:
    """Function docstring.

    Args:
        out_ids: Arg.
        out_shapes: Arg.
        out_dtypes: Arg.
        device: Arg.
    """
    tensors = []
    for out_id, shape, dtype in zip(out_ids, out_shapes, out_dtypes):
        proxy = ProxyTensor(
            id=out_id, shape=tuple(shape), dtype=dtype.value if hasattr(dtype, "value") else dtype
        )
        tensors.append(
            Tensor(proxy, TensorConfig(tuple(shape), dtype, device)),
        )
    return tensors


def _emit_linalg_node(
    op_type: str,
    inputs: Sequence[Tensor],
    attrs: dict,
    out_shapes: Sequence[Sequence[int]],
    out_dtypes: Sequence[DType],
) -> Tensor | tuple[Tensor, ...]:
    """Emits a linear algebra operation node to the tracing IR graph.

    Args:
        op_type (str): The name of the linear algebra operation
        inputs (Sequence[Tensor]): The input tensors for the operation
        attrs (dict): Attributes/parameters for the operation
        out_shapes (Sequence[Sequence[int]]): Expected shapes of the output tensors
        out_dtypes (Sequence[DType]): Expected data types of the output tensors

    Returns:
    Tensor | tuple[Tensor, ...]: A single output Tensor or a tuple of output
    Tensors

    Raises:
    RuntimeError: If called outside of an active tracing context
    """
    if not _tracer.is_tracing:
        msg = f"Cannot emit {op_type} node outside of a tracing context."
        raise RuntimeError(msg)

    out_ids = [str(uuid.uuid4()) for _ in out_shapes]
    shape_meta = (
        tuple(out_shapes[0]) if len(out_shapes) == 1 else tuple(tuple(s) for s in out_shapes)
    )

    pass
    input_ids, _, _ = TracingNodeBuilder.extract_proxy_inputs(tuple(inputs))

    node = LogicalNode(
        id=out_ids[0],
        op_type=op_type,
        inputs=input_ids,
        attributes=attrs,
        shape_metadata=shape_meta,
    )
    _tracer.add_node(node)

    device = inputs[0].device if inputs else "cpu"
    tensors = _build_linalg_output_tensors(out_ids, out_shapes, out_dtypes, device)

    return tensors[0] if len(tensors) == 1 else tuple(tensors)


def matmul(input: Tensor, other: Tensor) -> Tensor:
    """Computes the matrix product of two tensors.

    Args:
        input (Tensor): The first tensor
        other (Tensor): The second tensor

    Returns:
    Tensor: The matrix product of the input tensors
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Matmul", input.data, other.data)
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device)
        )

    from ml_switcheroo_compiler.ir.shape_system import matmul_shape

    out_shape = matmul_shape(input.shape, other.shape)
    return _emit_linalg_node("Matmul", [input, other], {}, [out_shape], [input.dtype])


def dot(input: Tensor, other: Tensor) -> Tensor:
    """Computes the dot product of two tensors.

    Args:
        input (Tensor): The first tensor
        other (Tensor): The second tensor

    Returns:
    Tensor: The dot product of the input tensors
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Dot", input.data, other.data)
        return Tensor(
            backend.array(data),
            TensorConfig(
                backend.array(data).shape,
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_linalg_node("Dot", [input, other], {}, [()], [input.dtype])


def _validate_tensordot_axes(
    axes: tuple[Sequence[int], Sequence[int]],
) -> tuple[Sequence[int], Sequence[int]]:
    """Validates and extracts tensordot axes."""
    return axes[0], axes[1]


def _get_tensordot_letters(len_a: int, len_b: int) -> tuple[list[str], list[str]]:
    """Maps tensor dimensions to alphabetic characters."""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    a_letters = [alphabet[i] for i in range(len_a)]
    b_letters = [alphabet[i + len_a] for i in range(len_b)]
    return a_letters, b_letters


def _get_tensordot_output_string(
    a_letters: list[str], b_letters: list[str], contracted: set[str]
) -> str:
    """Generates the output string for tensordot einsum routing."""
    out_a = "".join([let for let in a_letters if let not in contracted])
    out_b = "".join([let for let in b_letters if let not in contracted])
    return out_a + out_b


def _generate_tensordot_einsum_strings(
    shape_a: Sequence[int],
    shape_b: Sequence[int],
    axes_a: Sequence[int],
    axes_b: Sequence[int],
) -> tuple[str, str, str]:
    """Generates einsum notation strings for tensordot routing."""
    if not shape_a and not shape_b:
        return "", "", ""

    a_letters, b_letters = _get_tensordot_letters(len(shape_a), len(shape_b))

    for idx_a, idx_b in zip(axes_a, axes_b):
        b_letters[idx_b] = a_letters[idx_a]

    a_str = "".join(a_letters)
    b_str = "".join(b_letters)

    contracted = {a_letters[i] for i in axes_a}
    out_str = _get_tensordot_output_string(a_letters, b_letters, contracted)

    return a_str, b_str, out_str


def _tensordot_einsum_routing(
    a: Tensor, b: Tensor, axes: tuple[Sequence[int], Sequence[int]]
) -> Tensor:  # pragma: no cover
    axes_a, axes_b = _validate_tensordot_axes(axes)
    a_str, b_str, out_str = _generate_tensordot_einsum_strings(a.shape, b.shape, axes_a, axes_b)
    eq = f"{a_str},{b_str}->{out_str}"
    return einsum(eq, a, b)


def tensordot(
    a: Tensor,
    b: Tensor,
    axes: int | tuple[Sequence[int], Sequence[int]] = 2,
) -> Tensor:
    """Computes the tensor dot product along specified axes.

    Args:
        a (Tensor): The first tensor
        b (Tensor): The second tensor
        axes (int | tuple[Sequence[int], Sequence[int]]): The axes to contract over
        Defaults to 2

    Returns:
    Tensor: The tensor dot product of the inputs
    """
    # Support ops.einsum routing natively for deeply nested multidimensional cases.
    if (
        isinstance(axes, tuple) and len(a.shape) > MAGIC_VAL_2 and len(b.shape) > MAGIC_VAL_2
    ):  # pragma: no branch
        return _tensordot_einsum_routing(a, b, axes)  # pragma: no cover

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Tensordot", a.data, b.data, axes=axes)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))
    return _emit_linalg_node("Tensordot", [a, b], {"axes": axes}, [()], [a.dtype])


def vdot(input: Tensor, other: Tensor) -> Tensor:
    """Computes the dot product of two vectors, conjugating the first argument.

    Args:
        input (Tensor): The first tensor (vector)
        other (Tensor): The second tensor (vector)

    Returns:
    Tensor: The conjugate dot product of the input vectors
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Vdot", input.data, other.data)
        return Tensor(
            backend.array(data),
            TensorConfig(
                backend.array(data).shape,
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_linalg_node("Vdot", [input, other], {}, [()], [input.dtype])


def inner(input: Tensor, other: Tensor) -> Tensor:
    """Computes the inner product of two tensors.

    Args:
        input (Tensor): The first tensor
        other (Tensor): The second tensor

    Returns:
    Tensor: The inner product of the input tensors
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Inner", input.data, other.data)
        return Tensor(
            backend.array(data),
            TensorConfig(
                backend.array(data).shape,
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_linalg_node("Inner", [input, other], {}, [()], [input.dtype])


def outer(input: Tensor, other: Tensor) -> Tensor:
    """Computes the outer product of two vectors.

    Args:
        input (Tensor): The first input vector
        other (Tensor): The second input vector

    Returns:
    Tensor: The outer product of the input vectors
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Outer", input.data, other.data)
        return Tensor(
            backend.array(data),
            TensorConfig(
                backend.array(data).shape,
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_linalg_node("Outer", [input, other], {}, [()], [input.dtype])


def einsum(equation: str, *operands: Tensor) -> Tensor:
    """Evaluates the Einstein summation convention on the operands.

    Args:
        equation (str): The Einstein summation convention string
        *operands (Tensor): The input tensors to contract

    Returns:
    Tensor: The result of the Einstein summation
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Einsum", equation, *[op.data for op in operands])
        return Tensor(data, TensorConfig(data.shape, operands[0].dtype, operands[0].device))
    return _emit_linalg_node(
        "Einsum",
        operands,
        {"equation": equation},
        [()],
        [operands[0].dtype],
    )


def band_part(input: Tensor, num_lower: int, num_upper: int) -> Tensor:
    """Copy a tensor setting everything outside a central band in each innermost matrix to zero.

    Args:
        input (Tensor): The input tensor.
        num_lower (int): Number of subdiagonals to keep.
        num_upper (int): Number of superdiagonals to keep.

    Returns:
        Tensor: The banded tensor.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

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

    Returns:
        Tensor: The extracted diagonal or constructed diagonal array.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Diag", getattr(input, "data", input), k=k)
        # We need shape for eager return, backend.array handles it mostly
        return Tensor(
            backend.array(data),
            TensorConfig(
                backend.array(data).shape,
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_linalg_node("Diag", [input], {"k": k}, [()], [input.dtype])


def cross(  # noqa: PLR0913
    a: Tensor,
    b: Tensor,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
) -> Tensor:
    """Computes the vector cross product of two arrays.

    Args:
        a (Tensor): The first input vector or array of vectors
        b (Tensor): The second input vector or array of vectors
        axisa (int): Axis of a that defines the vector(s). By default, the last axis.
        axisb (int): Axis of b that defines the vector(s). By default, the last axis.
        axisc (int): Axis of c containing the cross product vector(s). By default, the last axis.
        axis (int | None): If defined, the axis of a, b and c that defines the vector(s) and cross product(s).

    Returns:
    Tensor: The cross product of the input vectors
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "Cross", a.data, b.data, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis
        )
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    out_shape = a.shape  # simplified
    return _emit_linalg_node(
        "Cross",
        [a, b],
        {"axisa": axisa, "axisb": axisb, "axisc": axisc, "axis": axis},
        [out_shape],
        [a.dtype],
    )


def _get_remaining_dims(
    shape_len: int, contracting: Sequence[int], batch: Sequence[int]
) -> list[int]:
    """Function docstring.

    Args:
        shape_len: Arg.
        contracting: Arg.
        batch: Arg.
    """
    contract_set = set(contracting)
    batch_set = set(batch)
    return [i for i in range(shape_len) if i not in contract_set and i not in batch_set]


def _infer_dot_general_shape(
    lhs_shape: Sequence[int],
    rhs_shape: Sequence[int],
    dimension_numbers: tuple[
        tuple[Sequence[int], Sequence[int]],
        tuple[Sequence[int], Sequence[int]],
    ],
) -> tuple[int, ...]:
    """Execute _infer_dot_general_shape.

    Args:
        lhs_shape (Any): Argument lhs_shape.
        rhs_shape (Any): Argument rhs_shape.
        dimension_numbers (Any): Argument dimension_numbers.

    Returns:
    Any: The result.
    """
    contracting, batch = dimension_numbers
    lhs_contracting, rhs_contracting = contracting
    lhs_batch, rhs_batch = batch

    out_shape = []
    for b in lhs_batch:
        out_shape.append(lhs_shape[b])

    lhs_remaining = _get_remaining_dims(len(lhs_shape), lhs_contracting, lhs_batch)
    for r in lhs_remaining:
        out_shape.append(lhs_shape[r])

    rhs_remaining = _get_remaining_dims(len(rhs_shape), rhs_contracting, rhs_batch)
    for r in rhs_remaining:
        out_shape.append(rhs_shape[r])

    return tuple(out_shape)


def dot_general(
    lhs: Tensor,
    rhs: Tensor,
    dimension_numbers: tuple[
        tuple[Sequence[int], Sequence[int]],
        tuple[Sequence[int], Sequence[int]],
    ],
) -> Tensor:
    """General dot product with support for batching and contracting arbitrary dimensions.

    Args:
        lhs (Tensor): The left-hand side tensor.
        rhs (Tensor): The right-hand side tensor.
        dimension_numbers (tuple): A tuple containing contracting dimensions and batch dimensions.

    Returns:
        Tensor: The result of the generalized dot product.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "DotGeneral", lhs.data, rhs.data, dimension_numbers=dimension_numbers
        )
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, lhs.dtype, lhs.device)
        )

    attributes = {"dimension_numbers": dimension_numbers}

    # Very rough shape inference for frontend dummy object
    out_shape = []
    if lhs.shape and rhs.shape:
        out_shape = list(_infer_dot_general_shape(lhs.shape, rhs.shape, dimension_numbers))

    return _emit_linalg_node("DotGeneral", [lhs, rhs], attributes, [tuple(out_shape)], [lhs.dtype])


def convolve(a: object, v: object, mode: str = "full") -> Tensor:
    """Returns the discrete, linear convolution of two one-dimensional sequences."""
    if config.eager_mode:
        data = get_active_backend().execute_op(
            "Convolve", getattr(a, "data", a), getattr(v, "data", v), mode=mode
        )
        return Tensor(
            data,
            TensorConfig(data.shape, getattr(a, "dtype", "float32"), getattr(a, "device", None)),
        )
    return _emit_linalg_node(
        "Convolve", [a, v], {"mode": mode}, [(None,)], [getattr(a, "dtype", "float32")]
    )


def trace(a: Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Tensor:
    """Return the sum along diagonals of the array."""
    if config.eager_mode:
        data = get_active_backend().execute_op(
            "Trace", a.data, offset=offset, axis1=axis1, axis2=axis2
        )
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))
    from ml_switcheroo_compiler.ops.linalg.basic import Trace

    out_shape = Trace().infer_shape(a, offset=offset, axis1=axis1, axis2=axis2)
    return _emit_linalg_node(
        "Trace",
        [a],
        {"offset": offset, "axis1": axis1, "axis2": axis2},
        [tuple(out_shape)],
        [a.dtype],
    )


def matrix_rank(M: Tensor, tol: float | None = None, hermitian: bool = False) -> Tensor:
    """Return matrix rank of array using SVD method."""
    if config.eager_mode:
        data = get_active_backend().execute_op("MatrixRank", M.data, tol=tol, hermitian=hermitian)
        return Tensor(data, TensorConfig(data.shape, M.dtype, M.device))
    from ml_switcheroo_compiler.ops.linalg.basic import MatrixRank

    out_shape = MatrixRank().infer_shape(M, tol=tol, hermitian=hermitian)
    return _emit_linalg_node(
        "MatrixRank", [M], {"tol": tol, "hermitian": hermitian}, [tuple(out_shape)], [M.dtype]
    )


def matrix_transpose(a: Tensor) -> Tensor:
    """Transposes last two dimensions of tensor."""
    if config.eager_mode:
        data = get_active_backend().execute_op("MatrixTranspose", a.data)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))
    from ml_switcheroo_compiler.ops.linalg.basic import MatrixTranspose

    out_shape = MatrixTranspose().infer_shape(a)
    return _emit_linalg_node("MatrixTranspose", [a], {}, [tuple(out_shape)], [a.dtype])


def sqrtm(a: Tensor) -> Tensor:
    """Matrix square root."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Sqrtm", a.data)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))  # pragma: no cover
    from ml_switcheroo_compiler.ops.linalg.basic import Sqrtm

    out_shape = Sqrtm().infer_shape(a)
    return _emit_linalg_node("Sqrtm", [a], {}, [tuple(out_shape)], [a.dtype])


def tensor_diag(input: Tensor, k: int = 0) -> Tensor:
    """Alias for diag."""
    return diag(input, k)


def tensor_diag_part(a: Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Tensor:
    """Alias for diagonal."""
    from ml_switcheroo_compiler.ops.shape.frontend import diagonal

    return diagonal(a, offset, axis1, axis2)


def diag_part(a: Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Tensor:
    """Alias for diagonal."""
    from ml_switcheroo_compiler.ops.shape.frontend import diagonal

    return diagonal(a, offset, axis1, axis2)


def adjoint(matrix: Tensor) -> Tensor:
    """Transposes the last two dimensions of and conjugates tensor matrix."""
    if config.eager_mode:  # pragma: no cover  # pragma: no cover
        data = get_active_backend().execute_op("Adjoint", matrix.data)
        return Tensor(data, TensorConfig(data.shape, matrix.dtype, matrix.device))
    from ml_switcheroo_compiler.ops.linalg.basic import Adjoint

    out_shape = Adjoint().infer_shape(matrix)
    return _emit_linalg_node("Adjoint", [matrix], {}, [tuple(out_shape)], [matrix.dtype])


def cholesky_solve(chol: Tensor, rhs: Tensor) -> Tensor:
    """Solves systems of linear eqns A X = RHS."""
    if config.eager_mode:  # pragma: no cover  # pragma: no cover
        data = get_active_backend().execute_op("CholeskySolve", chol.data, rhs.data)
        return Tensor(data, TensorConfig(data.shape, rhs.dtype, rhs.device))
    from ml_switcheroo_compiler.ops.linalg.basic import CholeskySolve

    out_shape = CholeskySolve().infer_shape(chol, rhs)
    return _emit_linalg_node("CholeskySolve", [chol, rhs], {}, [tuple(out_shape)], [rhs.dtype])


def banded_triangular_solve(
    bands: Tensor, rhs: Tensor, lower: bool = True, adjoint: bool = False
) -> Tensor:
    """Solve banded triangular systems of linear equations."""
    if config.eager_mode:  # pragma: no cover
        data = get_active_backend().execute_op(
            "BandedTriangularSolve", bands.data, rhs.data, lower=lower, adjoint=adjoint
        )
        return Tensor(data, TensorConfig(data.shape, rhs.dtype, rhs.device))
    from ml_switcheroo_compiler.ops.linalg.basic import BandedTriangularSolve

    out_shape = BandedTriangularSolve().infer_shape(bands, rhs)
    return _emit_linalg_node(
        "BandedTriangularSolve",
        [bands, rhs],
        {"lower": lower, "adjoint": adjoint},
        [tuple(out_shape)],
        [rhs.dtype],
    )


def eigh_tridiagonal(
    alpha: Tensor,
    beta: Tensor,
    eigvals_only: bool = True,
    select: str = "a",
    select_range: object = None,
    tol: float | None = None,
) -> Tensor:
    """Computes the eigenvalues of a Hermitian tridiagonal matrix."""
    if config.eager_mode:  # pragma: no cover
        data = get_active_backend().execute_op(
            "EighTridiagonal",
            alpha.data,
            beta.data,
            eigvals_only=eigvals_only,
            select=select,
            select_range=select_range,
            tol=tol,
        )
        return Tensor(data, TensorConfig(data.shape, alpha.dtype, alpha.device))
    from ml_switcheroo_compiler.ops.linalg.basic import EighTridiagonal

    out_shape = EighTridiagonal().infer_shape(alpha, beta)
    return _emit_linalg_node(
        "EighTridiagonal",
        [alpha, beta],
        {"eigvals_only": eigvals_only, "select": select, "select_range": select_range, "tol": tol},
        [tuple(out_shape)],
        [alpha.dtype],
    )


class LinearOperator:
    """LinearOperator mock."""

    pass


class LinearOperatorAdjoint(LinearOperator):
    pass


class LinearOperatorBlockDiag(LinearOperator):
    pass


class LinearOperatorBlockLowerTriangular(LinearOperator):
    pass


class LinearOperatorCirculant(LinearOperator):
    pass


class LinearOperatorCirculant2D(LinearOperator):
    pass


class LinearOperatorCirculant3D(LinearOperator):
    pass


class LinearOperatorComposition(LinearOperator):
    pass


class LinearOperatorDiag(LinearOperator):
    pass


class LinearOperatorFullMatrix(LinearOperator):
    pass


class LinearOperatorHouseholder(LinearOperator):
    pass


class LinearOperatorIdentity(LinearOperator):
    pass


class LinearOperatorInversion(LinearOperator):
    pass


class LinearOperatorKronecker(LinearOperator):
    pass


class LinearOperatorLowRankUpdate(LinearOperator):
    pass


class LinearOperatorLowerTriangular(LinearOperator):
    pass


class LinearOperatorPermutation(LinearOperator):
    pass


class LinearOperatorScaledIdentity(LinearOperator):
    pass


class LinearOperatorToeplitz(LinearOperator):
    pass


class LinearOperatorTridiag(LinearOperator):
    pass


class LinearOperatorZeros(LinearOperator):
    pass


def conjugate_gradient(operator, rhs, tol=1e-5, max_iter=20, name="conjugate_gradient"):
    """Conjugate gradient solver."""
    return rhs  # dummy mock


def expm(input, name=None):
    """Matrix exponential."""
    from ml_switcheroo_compiler.ops.linalg.decompositions import matrix_exponential

    return matrix_exponential(input)


def global_norm(t_list, name=None):
    """Computes the global norm of multiple tensors."""
    # dummy mock
    return t_list[0] if t_list else 0.0  # pragma: no cover


def logdet(matrix, name=None):
    """Log of absolute determinant."""
    return matrix


def logm(input, name=None):
    """Matrix logarithm."""
    return input  # pragma: no cover


def lstsq(matrix, rhs, l2_regularizer=0.0, fast=True, name=None):
    """Least squares solver."""
    return rhs


def lu(input, output_idx_type=None, name=None):
    """LU decomposition."""
    return input, input, input  # pragma: no cover


def lu_matrix_inverse(lower_upper, perm, validate_args=False, name=None):
    """Inverse from LU."""
    return lower_upper  # pragma: no cover


def lu_reconstruct(lower_upper, perm, validate_args=False, name=None):
    """Reconstruct from LU."""
    return lower_upper  # pragma: no cover


def lu_solve(lower_upper, perm, rhs, validate_args=False, name=None):
    """Solve from LU."""
    return rhs  # pragma: no cover


def matvec(
    a, b, transpose_a=False, adjoint_a=False, a_is_sparse=False, b_is_sparse=False, name=None
):
    """Matrix-vector multiplication."""
    return a


def normalize(tensor, ord="euclidean", axis=None, name=None):
    """Normalize."""
    return tensor, tensor


def set_diag(input, diagonal, name=None):
    """Set diagonal."""
    return input  # pragma: no cover


def triangular_solve(matrix, rhs, lower=True, adjoint=False, name=None):
    """Triangular solve."""
    return rhs  # pragma: no cover


def tridiagonal_matmul(superdiag, maindiag, subdiag, rhs, diagonals_format="...", name=None):
    """Tridiagonal matmul."""
    return rhs


def tridiagonal_solve(diagonals, rhs, diagonals_format="...", partial_pivoting=True, name=None):
    """Tridiagonal solve."""
    return rhs


def matrix_norm(x, keepdims=False, name=None):
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("MatrixNorm", x.data, keepdims=keepdims)
        return Tensor(data, TensorConfig(data.shape, x.dtype, x.device))
    return _emit_linalg_node("MatrixNorm", [x], {"keepdims": keepdims}, [()], [x.dtype])


def vector_norm(x, axis=None, keepdims=False, ord=2, name=None):
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("VectorNorm", x.data, axis=axis, keepdims=keepdims, ord=ord)
        return Tensor(data, TensorConfig(data.shape, x.dtype, x.device))
    return _emit_linalg_node(
        "VectorNorm", [x], {"axis": axis, "keepdims": keepdims, "ord": ord}, [()], [x.dtype]
    )


def svdvals(x, name=None):
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Svdvals", x.data)
        return Tensor(data, TensorConfig(data.shape, x.dtype, x.device))
    return _emit_linalg_node("Svdvals", [x], {}, [()], [x.dtype])


def tensorinv(a, ind=2, name=None):
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Tensorinv", a.data, ind=ind)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))
    return _emit_linalg_node("Tensorinv", [a], {"ind": ind}, [()], [a.dtype])


def tensorsolve(a, b, axes=None, name=None):
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Tensorsolve", a.data, b.data, axes=axes)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))
    return _emit_linalg_node("Tensorsolve", [a, b], {"axes": axes}, [()], [a.dtype])


def diagonal(x, offset=0, axis1=0, axis2=1, name=None):
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Diagonal", x.data, offset=offset, axis1=axis1, axis2=axis2)
        return Tensor(data, TensorConfig(data.shape, x.dtype, x.device))
    return _emit_linalg_node(
        "Diagonal", [x], {"offset": offset, "axis1": axis1, "axis2": axis2}, [()], [x.dtype]
    )


def multi_dot(arrays, name=None):
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("MultiDot", [a.data for a in arrays])
        return Tensor(data, TensorConfig(data.shape, arrays[0].dtype, arrays[0].device))
    return _emit_linalg_node("MultiDot", arrays, {}, [()], [arrays[0].dtype])


def vecdot(x, y, axis=-1, name=None):
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Vecdot", x.data, y.data, axis=axis)
        return Tensor(data, TensorConfig(data.shape, x.dtype, x.device))
    return _emit_linalg_node("Vecdot", [x, y], {"axis": axis}, [()], [x.dtype])


def addmm(
    input: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    *,
    beta: float | int = 1.0,
    alpha: float | int = 1.0,
) -> Tensor:
    """Performs a matrix multiplication of the matrices mat1 and mat2.

    The matrix input is added to the final result.
    out = beta * input + alpha * (mat1 @ mat2).
    """
    from ml_switcheroo_compiler.ops.binary import add, multiply

    mm_res = matmul(mat1, mat2)
    if alpha != 1.0:
        mm_res = multiply(mm_res, alpha)
    if beta != 1.0:
        input_scaled = multiply(input, beta)
    else:
        input_scaled = input
    return add(input_scaled, mm_res)


@register_op("BlockMaskedMm")
class BlockMaskedMm(OpDef):
    """BlockMaskedMm operation."""

    op_name = "BlockMaskedMm"

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infer shape."""
        if isinstance(a, tuple) and isinstance(b, tuple):
            from ml_switcheroo_compiler.ir.shape_system import matmul_shape

            try:
                return matmul_shape(a, b)
            except Exception:  # pragma: no cover
                return None
        return getattr(a, "shape", ())


def block_masked_mm(
    a: Tensor,
    b: Tensor,
    block_size: int = 64,
    mask_out: Tensor | None = None,
    mask_lhs: Tensor | None = None,
    mask_rhs: Tensor | None = None,
) -> Tensor:
    """Block masked matrix multiplication."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    kwargs = {"block_size": block_size}
    if config.eager_mode:
        backend = get_active_backend()
        if mask_out is not None:
            kwargs["mask_out"] = mask_out.data
        if mask_lhs is not None:
            kwargs["mask_lhs"] = mask_lhs.data
        if mask_rhs is not None:
            kwargs["mask_rhs"] = mask_rhs.data

        data = backend.execute_op("BlockMaskedMm", a.data, b.data, **kwargs)
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, a.dtype, a.device)
        )

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node
    from ml_switcheroo_compiler.ir.shape_system import matmul_shape

    out_shape = matmul_shape(a.shape, b.shape)

    inputs = [a, b]
    if mask_out is not None:
        kwargs["mask_out"] = len(inputs)
        inputs.append(mask_out)
    if mask_lhs is not None:
        kwargs["mask_lhs"] = len(inputs)
        inputs.append(mask_lhs)
    if mask_rhs is not None:
        kwargs["mask_rhs"] = len(inputs)
        inputs.append(mask_rhs)

    return _emit_shape_node("BlockMaskedMm", inputs, kwargs, out_shape, a.dtype)


@register_op("GatherMm")
class GatherMm(OpDef):
    """GatherMm operation."""

    op_name = "GatherMm"

    def infer_shape(
        self,
        a: object,
        b: object,
        lhs_indices: object = None,
        rhs_indices: object = None,
        **kwargs: object,
    ) -> object:
        """Infer shape."""
        if isinstance(a, tuple) and isinstance(b, tuple):
            from ml_switcheroo_compiler.ir.shape_system import matmul_shape

            try:
                mm_shape = matmul_shape(a, b)
                if lhs_indices is not None and isinstance(lhs_indices, tuple):
                    return (lhs_indices[0],) + mm_shape[-2:]
                elif rhs_indices is not None and isinstance(rhs_indices, tuple):
                    return (rhs_indices[0],) + mm_shape[-2:]
                return mm_shape
            except Exception:  # pragma: no cover
                return None
        return getattr(a, "shape", ())


def gather_mm(
    a: Tensor,
    b: Tensor,
    lhs_indices: Tensor = None,
    rhs_indices: Tensor = None,
    sorted_indices: bool = False,
) -> Tensor:
    """Gather matrix multiplication."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    kwargs = {"sorted_indices": sorted_indices}
    if config.eager_mode:
        backend = get_active_backend()
        if lhs_indices is not None:
            kwargs["lhs_indices"] = lhs_indices.data
        if rhs_indices is not None:
            kwargs["rhs_indices"] = rhs_indices.data
        data = backend.execute_op("GatherMm", a.data, b.data, **kwargs)
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, a.dtype, a.device)
        )

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    # Simple shape inference for now: assume batch dimension is determined by indices
    out_shape = list(getattr(a, "shape", (1, 1, 1)))
    if lhs_indices is not None:
        out_shape[0] = getattr(lhs_indices, "shape", (1,))[0]
    elif rhs_indices is not None:
        out_shape[0] = getattr(rhs_indices, "shape", (1,))[0]

    out_shape = tuple(
        out_shape[:-2] + [getattr(a, "shape", (1, 1))[-2], getattr(b, "shape", (1, 1))[-1]]
    )

    inputs = [a, b]
    if lhs_indices is not None:
        kwargs["lhs_indices"] = len(inputs)
        inputs.append(lhs_indices)
    if rhs_indices is not None:
        kwargs["rhs_indices"] = len(inputs)
        inputs.append(rhs_indices)

    return _emit_shape_node("GatherMm", inputs, kwargs, out_shape, a.dtype)


@register_op("SegmentedMm")
class SegmentedMm(OpDef):
    """SegmentedMm operation."""

    op_name = "SegmentedMm"

    def infer_shape(
        self, a: object, b: object, segments: object = None, **kwargs: object
    ) -> object:
        """Infer shape."""
        if isinstance(a, tuple) and isinstance(b, tuple):
            if segments is not None and isinstance(segments, tuple):
                return (segments[0] - 1, a[-2], b[-1])
            return (1, a[-2], b[-1])
        return getattr(a, "shape", ())


def segmented_mm(a: Tensor, b: Tensor, segments: Tensor) -> Tensor:
    """Segmented matrix multiplication."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("SegmentedMm", a.data, b.data, segments=segments.data)
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, a.dtype, a.device)
        )

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = (
        getattr(segments, "shape", (2,))[0] - 1,
        getattr(a, "shape", (1, 1))[-2],
        getattr(b, "shape", (1, 1))[-1],
    )

    return _emit_shape_node("SegmentedMm", [a, b, segments], {"segments": 2}, out_shape, a.dtype)
