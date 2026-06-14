"""Provides linear algebra operations for the ml_switcheroo_compiler framework.

This module contains standard linear algebra functions such as matrix multiplication,
decompositions (SVD, QR, Cholesky, LU), solvers, and other tensor operations. It
supports both eager execution using NumPy/SciPy and graph tracing by emitting logical
nodes to the intermediate representation (IR) graph
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.errors import UnimplementedMathError
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.tracing import ProxyTensor, _tracer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ml_switcheroo_compiler.core.dtype import DType


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
    # Simple broadcast fallback for shape_metadata
    shape_meta = (
        tuple(out_shapes[0]) if len(out_shapes) == 1 else tuple(tuple(s) for s in out_shapes)
    )

    node = LogicalNode(
        id=out_ids[0],
        op_type=op_type,
        inputs=[inp.data.id for inp in inputs],
        attributes=attrs,
        shape_metadata=shape_meta,
    )
    _tracer.add_node(node)

    tensors = []
    for _i, (out_id, shape, dtype) in enumerate(zip(out_ids, out_shapes, out_dtypes)):
        proxy = ProxyTensor(id=out_id, shape=tuple(shape), dtype=dtype.value)
        tensors.append(
            Tensor(data=proxy, shape=tuple(shape), dtype=dtype, device=inputs[0].device),
        )

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
        return Tensor(backend.array(data), backend.array(data).shape, input.dtype, input.device)
    return _emit_linalg_node("Matmul", [input, other], {}, [()], [input.dtype])


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
        return Tensor(backend.array(data), backend.array(data).shape, input.dtype, input.device)
    return _emit_linalg_node("Dot", [input, other], {}, [()], [input.dtype])


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
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Tensordot", a.data, b.data, axes=axes)
        return Tensor(data, data.shape, a.dtype, a.device)
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
        return Tensor(backend.array(data), backend.array(data).shape, input.dtype, input.device)
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
        return Tensor(backend.array(data), backend.array(data).shape, input.dtype, input.device)
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
        return Tensor(backend.array(data), backend.array(data).shape, input.dtype, input.device)
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
        return Tensor(data, data.shape, operands[0].dtype, operands[0].device)
    return _emit_linalg_node(
        "Einsum",
        operands,
        {"equation": equation},
        [()],
        [operands[0].dtype],
    )


def cholesky(input: Tensor) -> Tensor:
    """Computes the Cholesky decomposition of a symmetric/Hermitian positive-definite.

    matrix

    Args:
    input (Tensor): The input symmetric/Hermitian positive-definite matrix

    Returns:
    Tensor: The lower-triangular or upper-triangular Cholesky factor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Cholesky", input.data)
        return Tensor(backend.array(data), backend.array(data).shape, input.dtype, input.device)
    return _emit_linalg_node("Cholesky", [input], {}, [()], [input.dtype])


def svd(
    input: Tensor,
    full_matrices: bool = True,
    compute_uv: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes the Singular Value Decomposition (SVD) of a matrix.

    Args:
    input (Tensor): The input matrix of shape (..., M, N)
    full_matrices (bool): If True, matrices U and Vh have shapes (..., M, M)
        and (..., N, N). Otherwise, shapes are (..., M, K) and (..., K, N)
        where K = min(M, N). Defaults to True
    compute_uv (bool): Whether to compute U and Vh in addition to S. Defaults to
    True

    Returns:
    tuple[Tensor, Tensor, Tensor]: A tuple containing:
        - U (Tensor): Left singular vectors
        - S (Tensor): Singular values
        - Vh (Tensor): Right singular vectors (conjugate transposed)
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        u, s, vh = backend.execute_op(
            "Svd",
            input.data,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
        )
        return (
            Tensor(u, u.shape, input.dtype, input.device),
            Tensor(s, s.shape, input.dtype, input.device),
            Tensor(vh, vh.shape, input.dtype, input.device),
        )
    return _emit_linalg_node(
        "Svd",
        [input],
        {"full_matrices": full_matrices, "compute_uv": compute_uv},
        [(), (), ()],
        [input.dtype] * 3,
    )


def qr(input: Tensor, mode: str = "reduced") -> tuple[Tensor, Tensor]:
    """Computes the QR decomposition of a matrix.

    Args:
    input (Tensor): The input matrix
    mode (str): Specifies the mode of decomposition ('reduced', 'complete',
        'r', or 'raw'). Defaults to 'reduced'

    Returns:
    tuple[Tensor, Tensor]: A tuple containing:
        - Q (Tensor): The orthonormal matrix
        - R (Tensor): The upper-triangular matrix
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        q, r = backend.execute_op("Qr", input.data, mode=mode)
        return (
            Tensor(q, q.shape, input.dtype, input.device),
            Tensor(r, r.shape, input.dtype, input.device),
        )
    return _emit_linalg_node(
        "Qr",
        [input],
        {"mode": mode},
        [(), ()],
        [input.dtype] * 2,
    )


def inv(input: Tensor) -> Tensor:
    """Computes the multiplicative inverse of a square matrix.

    Args:
    input (Tensor): The square matrix to invert

    Returns:
    Tensor: The multiplicative inverse of the input matrix
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Inv", input.data)
        return Tensor(backend.array(data), backend.array(data).shape, input.dtype, input.device)
    return _emit_linalg_node("Inv", [input], {}, [()], [input.dtype])


def pinv(input: Tensor, rcond: float = 1e-15) -> Tensor:
    """Computes the Moore-Penrose pseudo-inverse of a matrix.

    Args:
    input (Tensor): The matrix to invert
    rcond (float): Cutoff for small singular values. Defaults to 1e-15

    Returns:
    Tensor: The pseudo-inverse of the input matrix
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Pinv", input.data, rcond=rcond)
        return Tensor(data, data.shape, input.dtype, input.device)  # pragma: no cover
    return _emit_linalg_node(
        "Pinv", [input], {"rcond": rcond}, [()], [input.dtype]
    )  # pragma: no cover


def det(input: Tensor) -> Tensor:
    """Computes the determinant of a square matrix.

    Args:
    input (Tensor): The square matrix

    Returns:
    Tensor: The determinant of the input matrix
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Det", input.data)
        return Tensor(backend.array(data), backend.array(data).shape, input.dtype, input.device)
    return _emit_linalg_node("Det", [input], {}, [()], [input.dtype])


def slogdet(input: Tensor) -> tuple[Tensor, Tensor]:
    """Computes the sign and natural logarithm of the determinant of a square matrix.

    Args:
    input (Tensor): The square matrix

    Returns:
    tuple[Tensor, Tensor]: A tuple containing:
        - sign (Tensor): A number representing the sign of the determinant
    - logdet (Tensor): The natural logarithm of the absolute value of the
    determinant
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        sign, logdet = backend.execute_op("Slogdet", input.data)
        return (
            Tensor(backend.array(sign), backend.array(sign).shape, input.dtype, input.device),
            Tensor(backend.array(logdet), backend.array(logdet).shape, input.dtype, input.device),
        )
    return _emit_linalg_node("Slogdet", [input], {}, [(), ()], [input.dtype] * 2)


def eigh(input: Tensor, UPLO: str = "L") -> tuple[Tensor, Tensor]:
    """Computes the eigenvalues and eigenvectors of a complex Hermitian or real symmetric.

    matrix

    Args:
    input (Tensor): The symmetric or Hermitian matrix
    UPLO (str): Specifies whether the calculation is done with the lower ('L')
        or upper ('U') triangular part of the matrix. Defaults to 'L'

    Returns:
    tuple[Tensor, Tensor]: A tuple containing:
        - eigenvalues (Tensor): The eigenvalues in ascending order
        - eigenvectors (Tensor): The column eigenvectors
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        w, v = backend.execute_op("Eigh", input.data, UPLO=UPLO)
        return (
            Tensor(w, w.shape, input.dtype, input.device),
            Tensor(v, v.shape, input.dtype, input.device),
        )
    return _emit_linalg_node(
        "Eigh",
        [input],
        {"UPLO": UPLO},
        [(), ()],
        [input.dtype] * 2,
    )


def eigvalsh(input: Tensor, UPLO: str = "L") -> Tensor:
    """Computes the eigenvalues of a complex Hermitian or real symmetric matrix.

    Args:
    input (Tensor): The symmetric or Hermitian matrix
    UPLO (str): Specifies whether the calculation is done with the lower ('L')
        or upper ('U') triangular part of the matrix. Defaults to 'L'

    Returns:
    Tensor: The eigenvalues in ascending order
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Eigvalsh", input.data, UPLO=UPLO)
        return Tensor(backend.array(data), backend.array(data).shape, input.dtype, input.device)
    return _emit_linalg_node(
        "Eigvalsh",
        [input],
        {"UPLO": UPLO},
        [()],
        [input.dtype],
    )


def matrix_power(input: Tensor, n: int) -> Tensor:
    """Raises a square matrix to the integer power `n`.

    Args:
    input (Tensor): The square matrix
    n (int): The exponent

    Returns:
    Tensor: The matrix raised to the power `n`
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("MatrixPower", input.data, n)
        return Tensor(data, data.shape, input.dtype, input.device)
    return _emit_linalg_node("MatrixPower", [input], {"n": n}, [()], [input.dtype])


def cross(
    a: object,
    b: object,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
) -> object:
    """Computes the vector cross product of two arrays.

    Args:
    a (object): The first input vector or array of vectors
    b (object): The second input vector or array of vectors
    axisa (int): Axis of `a` that defines the vector(s). Defaults to -1
    axisb (int): Axis of `b` that defines the vector(s). Defaults to -1
    axisc (int): Axis of the output that contains the cross product vector(s)
        Defaults to -1
    axis (int | None): If defined, the axis of `a`, `b`, and the output that
        defines the vector(s). Defaults to None

    Returns:
    object: The cross product of the input vectors
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op("Cross", a, b, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)


def solve(a: object, b: object) -> object:
    """Solves a linear matrix equation, or system of linear scalar equations.

    Args:
    a (object): Coefficient matrix
    b (object): Ordinate or 'dependent variable' values

    Returns:
    object: Solution to the system of linear equations
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op(
        "Solve",
        (a.data if hasattr(a, "device") else a),
        (b.data if hasattr(b, "device") else b),
    )


def solve_triangular(
    a: object,
    b: object,
    trans: int = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> object:
    """Solves the equation `a x = b` for `x`, assuming `a` is a triangular matrix.

    Args:
    a (object): Triangular coefficient matrix
    b (object): Right-hand side matrix or vector
    trans (int): Type of system to solve. 0 for `a x = b`, 1 for `a^T x = b`,
        2 for `a^H x = b`. Defaults to 0
    lower (bool): Use only the lower triangular part of `a`. If False, use the
        upper triangular part. Defaults to False
    unit_diagonal (bool): If True, diagonal elements of `a` are assumed to be 1
        Defaults to False
    overwrite_b (bool): Allow overwriting data in `b` for speed. Defaults to False
    check_finite (bool): Whether to check that the input matrices contain only
        finite numbers. Defaults to True

    Returns:
    object: The solution matrix `x`
    """
    import scipy.linalg as spla

    return spla.solve_triangular(
        a,
        b,
        trans=trans,
        lower=lower,
        unit_diagonal=unit_diagonal,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
    )


def lu(
    a: object,
    permute_l: bool = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> object:
    """Computes the LU decomposition of a matrix.

    Args:
    a (object): The input matrix to decompose
    permute_l (bool): If True, perform the multiplication P * L and return
        only PL and U. Defaults to False
    overwrite_a (bool): Allow overwriting data in `a` for speed. Defaults to False
    check_finite (bool): Whether to check that the input matrix contains only
        finite numbers. Defaults to True

    Returns:
    object: The LU decomposition components (P, L, U) or (PL, U) depending on
    `permute_l`
    """
    import scipy.linalg as spla

    return spla.lu(
        a,
        permute_l=permute_l,
        overwrite_a=overwrite_a,
        check_finite=check_finite,
    )


def lu_factor(
    a: object,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> object:
    """Computes pivoted LU decomposition of a matrix for use in `lu_solve`.

    Args:
    a (object): The input matrix to decompose
    overwrite_a (bool): Allow overwriting data in `a` for speed. Defaults to False
    check_finite (bool): Whether to check that the input matrix contains only
        finite numbers. Defaults to True

    Returns:
    object: A tuple (lu, piv) containing the LU factorization and pivot indices
    """
    import scipy.linalg as spla

    return spla.lu_factor(a, overwrite_a=overwrite_a, check_finite=check_finite)


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
        msg = "No direct numpy for dot_general"
        raise UnimplementedMathError(msg)

    attributes = {"dimension_numbers": dimension_numbers}

    # Very rough shape inference for frontend dummy object
    contracting, batch = dimension_numbers
    lhs_contracting, rhs_contracting = contracting
    lhs_batch, rhs_batch = batch

    out_shape = []
    if lhs.shape and rhs.shape:
        for b in lhs_batch:
            out_shape.append(lhs.shape[b])
        lhs_remaining = [
            i for i in range(len(lhs.shape)) if i not in lhs_contracting and i not in lhs_batch
        ]
        for r in lhs_remaining:
            out_shape.append(lhs.shape[r])
        rhs_remaining = [
            i for i in range(len(rhs.shape)) if i not in rhs_contracting and i not in rhs_batch
        ]
        for r in rhs_remaining:
            out_shape.append(rhs.shape[r])

    return _emit_linalg_node("DotGeneral", [lhs, rhs], attributes, [tuple(out_shape)], [lhs.dtype])


def conv_general_dilated(
    lhs: Tensor,
    rhs: Tensor,
    window_strides: Sequence[int],
    padding: Sequence[tuple[int, int]] | str,
    lhs_dilation: Sequence[int] | None = None,
    rhs_dilation: Sequence[int] | None = None,
    dimension_numbers: object = None,
) -> Tensor:
    """General N-dimensional convolution with support for strides, padding, and dilations.

    Args:
    lhs (Tensor): Left-hand side tensor (input).
    rhs (Tensor): Right-hand side tensor (filters/weights).
    window_strides (Sequence[int]): Strides of the window.
    padding (Sequence[tuple[int, int]] | str): Padding to apply.
    lhs_dilation (Sequence[int] | None): Dilation of the input.
    rhs_dilation (Sequence[int] | None): Dilation of the weights.
    dimension_numbers (object): Dimension numbers specification.

    Returns:
    Tensor: The result of the convolution.

    Raises:
    UnimplementedMathError: If called in eager mode.
    """
    if config.eager_mode:
        msg = "No direct numpy for conv_general_dilated"
        raise UnimplementedMathError(msg)

    inputs = [lhs, rhs]
    attributes = {
        "window_strides": window_strides,
        "padding": padding,
        "lhs_dilation": lhs_dilation,
        "rhs_dilation": rhs_dilation,
        "dimension_numbers": dimension_numbers,
    }

    from ml_switcheroo_compiler.ops.linalg.basic import ConvGeneralDilated

    op = ConvGeneralDilated()
    out_shape = op.infer_shape(
        lhs,
        rhs,
        window_strides,
        padding,
        lhs_dilation,
        rhs_dilation,
        dimension_numbers,
    )

    return _emit_linalg_node("ConvGeneralDilated", inputs, attributes, [out_shape], [lhs.dtype])


def fft(a: Tensor, n: int | None = None, axis: int = -1) -> Tensor:
    """Computes the one-dimensional discrete Fourier Transform.

    Args:
    a (Tensor): The input tensor
    n (int | None): Length of the transformed axis of the output
    axis (int): Axis over which to compute the FFT

    Returns:
    Tensor: The transformed tensor

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Fft", a.data, n=n, axis=axis)
        # Note: returning proper complex type is complex, using a mock DType mapping if possible
        # We will just return float32 here if complex not supported
        return Tensor(data, data.shape, a.dtype, a.device)

    from ml_switcheroo_compiler.ops.linalg.basic import Fft

    op = Fft()
    out_shape = op.infer_shape(a, n, axis)

    return _emit_linalg_node("Fft", [a], {"n": n, "axis": axis}, [out_shape], [a.dtype])


def rfft(a: Tensor, n: int | None = None, axis: int = -1) -> Tensor:
    """Computes the one-dimensional discrete Fourier Transform for real input.

    Args:
    a (Tensor): The input tensor
    n (int | None): Length of the transformed axis of the output
    axis (int): Axis over which to compute the FFT

    Returns:
    Tensor: The transformed tensor

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Rfft", a.data, n=n, axis=axis)
        return Tensor(data, data.shape, a.dtype, a.device)

    from ml_switcheroo_compiler.ops.linalg.basic import Rfft

    op = Rfft()
    out_shape = op.infer_shape(a, n, axis)

    return _emit_linalg_node("Rfft", [a], {"n": n, "axis": axis}, [out_shape], [a.dtype])
