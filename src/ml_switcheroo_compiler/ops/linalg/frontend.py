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

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
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
    tensors = []
    for out_id, shape, dtype in zip(out_ids, out_shapes, out_dtypes):
        proxy = ProxyTensor(id=out_id, shape=tuple(shape), dtype=dtype.value)
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

    from ml_switcheroo_compiler.ops.base import get_op

    op_def = get_op(op_type)()
    input_ids, _, _ = op_def._extract_proxy_inputs(tuple(inputs))

    node = LogicalNode(
        id=out_ids[0],
        op_type=op_type,
        inputs=input_ids,
        attributes=attrs,
        shape_metadata=shape_meta,
    )
    _tracer.add_node(node)

    tensors = _build_linalg_output_tensors(out_ids, out_shapes, out_dtypes, inputs[0].device)

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
        data = backend.execute_op("Diag", input.data, k=k)
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


def cross(
    a: object,
    b: object,
    **kwargs: object,
) -> object:
    """Computes the vector cross product of two arrays.

    Args:
        a (object): The first input vector or array of vectors
        b (object): The second input vector or array of vectors
        **kwargs (object): Optional arguments axisa, axisb, axisc, axis.

    Returns:
    object: The cross product of the input vectors
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    axisa = kwargs.get("axisa", -1)
    axisb = kwargs.get("axisb", -1)
    axisc = kwargs.get("axisc", -1)
    axis = kwargs.get("axis", None)
    return backend.execute_op("Cross", a, b, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)


def _get_remaining_dims(
    shape_len: int, contracting: Sequence[int], batch: Sequence[int]
) -> list[int]:
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
        "Convolve", [a, v], {"mode": mode}, (None,), getattr(a, "dtype", "float32")
    )
