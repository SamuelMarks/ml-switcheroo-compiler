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

    lhs_remaining = [
        i for i in range(len(lhs_shape)) if i not in lhs_contracting and i not in lhs_batch
    ]
    for r in lhs_remaining:
        out_shape.append(lhs_shape[r])

    rhs_remaining = [
        i for i in range(len(rhs_shape)) if i not in rhs_contracting and i not in rhs_batch
    ]
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
        msg = "No direct numpy for dot_general"
        raise UnimplementedMathError(msg)

    attributes = {"dimension_numbers": dimension_numbers}

    # Very rough shape inference for frontend dummy object
    out_shape = []
    if lhs.shape and rhs.shape:
        out_shape = list(_infer_dot_general_shape(lhs.shape, rhs.shape, dimension_numbers))

    return _emit_linalg_node("DotGeneral", [lhs, rhs], attributes, [tuple(out_shape)], [lhs.dtype])
