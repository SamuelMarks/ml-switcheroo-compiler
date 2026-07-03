"""Module docstring."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.shape_system import matmul_shape
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.binary import add, multiply
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

from .einsum_frontend import _infer_dot_general_shape
from .utils import _emit_linalg_node


def matmul(input: Tensor, other: Tensor) -> Tensor:
    """Computes the matrix product of two tensors.

    Args:
        input (Tensor): The first tensor
        other (Tensor): The second tensor

    Returns:
    Tensor: The matrix product of the input tensors
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Matmul", input.data, other.data)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))

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


def vdot(input: Tensor, other: Tensor) -> Tensor:
    """Computes the dot product of two vectors, conjugating the first argument.

    Args:
        input (Tensor): The first tensor (vector)
        other (Tensor): The second tensor (vector)

    Returns:
    Tensor: The conjugate dot product of the input vectors
    """
    if config.eager_mode:
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


def dot_general(
    lhs: Tensor,
    rhs: Tensor,
    dimension_numbers: tuple[tuple[Sequence[int], Sequence[int]], tuple[Sequence[int], Sequence[int]]],
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
        backend = get_active_backend()
        data = backend.execute_op("DotGeneral", lhs.data, rhs.data, dimension_numbers=dimension_numbers)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, lhs.dtype, lhs.device))
    attributes = {"dimension_numbers": dimension_numbers}
    out_shape = []
    if lhs.shape and rhs.shape:
        out_shape = list(_infer_dot_general_shape(lhs.shape, rhs.shape, dimension_numbers))
    return _emit_linalg_node("DotGeneral", [lhs, rhs], attributes, [tuple(out_shape)], [lhs.dtype])


def convolve(a: object, v: object, mode: str = "full") -> Tensor:
    """Returns the discrete, linear convolution of two one-dimensional sequences."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Convolve", getattr(a, "data", a), getattr(v, "data", v), mode=mode)
        return Tensor(
            data,
            TensorConfig(data.shape, getattr(a, "dtype", "float32"), getattr(a, "device", None)),
        )
    return _emit_linalg_node("Convolve", [a, v], {"mode": mode}, [(None,)], [getattr(a, "dtype", "float32")])


def matvec(a: object, b: object, transpose_a: object = False, adjoint_a: object = False, **kwargs: object) -> object:
    """Matrix-vector multiplication."""
    return a


def multi_dot(arrays: object, name: object = None) -> object:
    """Function docstring."""
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("MultiDot", [a.data for a in arrays])
        return Tensor(data, TensorConfig(data.shape, arrays[0].dtype, arrays[0].device))
    return _emit_linalg_node("MultiDot", arrays, {}, [()], [arrays[0].dtype])


def vecdot(x: object, y: object, axis: object = -1, name: object = None) -> object:
    """Function docstring."""
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Vecdot", x.data, y.data, axis=axis)
        return Tensor(data, TensorConfig(data.shape, x.dtype, x.device))
    return _emit_linalg_node("Vecdot", [x, y], {"axis": axis}, [()], [x.dtype])


def addmm(
    input: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    *,
    beta: (float | int) = 1.0,
    alpha: (float | int) = 1.0,
) -> Tensor:
    """Performs a matrix multiplication of the matrices mat1 and mat2.

    The matrix input is added to the final result.
    out = beta * input + alpha * (mat1 @ mat2).
    """
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
            try:
                return matmul_shape(a, b)
            except (ValueError, TypeError):
                return None
        return getattr(a, "shape", ())


def block_masked_mm(
    a: Tensor,
    b: Tensor,
    block_size: int = 64,
    masks: dict[str, Tensor | None] | None = None,
) -> Tensor:
    """Block masked matrix multiplication."""
    masks = masks or {}
    mask_out = masks.get("mask_out")
    mask_lhs = masks.get("mask_lhs")
    mask_rhs = masks.get("mask_rhs")

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
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, a.dtype, a.device))

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
            try:
                mm_shape = matmul_shape(a, b)
                if lhs_indices is not None and isinstance(lhs_indices, tuple):
                    return (lhs_indices[0],) + mm_shape[-2:]
                elif rhs_indices is not None and isinstance(rhs_indices, tuple):
                    return (rhs_indices[0],) + mm_shape[-2:]
                return mm_shape
            except (ValueError, TypeError):
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
    kwargs = {"sorted_indices": sorted_indices}
    if config.eager_mode:
        backend = get_active_backend()
        if lhs_indices is not None:
            kwargs["lhs_indices"] = lhs_indices.data
        if rhs_indices is not None:
            kwargs["rhs_indices"] = rhs_indices.data
        data = backend.execute_op("GatherMm", a.data, b.data, **kwargs)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, a.dtype, a.device))

    out_shape = list(getattr(a, "shape", (1, 1, 1)))
    if lhs_indices is not None:
        out_shape[0] = getattr(lhs_indices, "shape", (1,))[0]
    elif rhs_indices is not None:
        out_shape[0] = getattr(rhs_indices, "shape", (1,))[0]
    out_shape = tuple(out_shape[:-2] + [getattr(a, "shape", (1, 1))[-2], getattr(b, "shape", (1, 1))[-1]])
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

    def infer_shape(self, a: object, b: object, segments: object = None, **kwargs: object) -> object:
        """Infer shape."""
        if isinstance(a, tuple) and isinstance(b, tuple):
            if segments is not None and isinstance(segments, tuple):
                return segments[0] - 1, a[-2], b[-1]
            return 1, a[-2], b[-1]
        return getattr(a, "shape", ())


def segmented_mm(a: Tensor, b: Tensor, segments: Tensor) -> Tensor:
    """Segmented matrix multiplication."""
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("SegmentedMm", a.data, b.data, segments=segments.data)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, a.dtype, a.device))

    out_shape = (
        getattr(segments, "shape", (2,))[0] - 1,
        getattr(a, "shape", (1, 1))[-2],
        getattr(b, "shape", (1, 1))[-1],
    )
    return _emit_shape_node("SegmentedMm", [a, b, segments], {"segments": 2}, out_shape, a.dtype)
