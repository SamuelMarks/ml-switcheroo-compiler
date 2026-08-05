"""Core abstractions and logic definitions for matmul.py."""

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
    """Compute the matrix product of two tensors.

    Args:
        input (Tensor): The input parameter.
        other (Tensor): The other parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "Matmul",
            (input.data if type(input).__name__ == "Tensor" else input),
            (other.data if type(other).__name__ == "Tensor" else other),
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, getattr(input, "dtype", None), getattr(input, "device", None)),
        )
    out_shape = matmul_shape(input.shape, other.shape)
    return _emit_linalg_node("Matmul", [input, other], {}, [out_shape], [getattr(input, "dtype", None)])


def dot(input: Tensor, other: Tensor) -> Tensor:
    """Compute the dot product of two tensors.

    Args:
        input (Tensor): The input parameter.
        other (Tensor): The other parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "Dot",
            (input.data if type(input).__name__ == "Tensor" else input),
            (other.data if type(other).__name__ == "Tensor" else other),
        )
        return Tensor(
            backend.array(data),
            TensorConfig(
                backend.array(data).shape,
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_linalg_node("Dot", [input, other], {}, [()], [getattr(input, "dtype", None)])


def vdot(input: Tensor, other: Tensor) -> Tensor:
    """Compute the dot product of two vectors, conjugating the first argument.

    Args:
        input (Tensor): The input parameter.
        other (Tensor): The other parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "Vdot",
            (input.data if type(input).__name__ == "Tensor" else input),
            (other.data if type(other).__name__ == "Tensor" else other),
        )
        return Tensor(
            backend.array(data),
            TensorConfig(
                backend.array(data).shape,
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_linalg_node("Vdot", [input, other], {}, [()], [getattr(input, "dtype", None)])


def inner(input: Tensor, other: Tensor) -> Tensor:
    """Compute the inner product of two tensors.

    Args:
        input (Tensor): The input parameter.
        other (Tensor): The other parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "Inner",
            (input.data if type(input).__name__ == "Tensor" else input),
            (other.data if type(other).__name__ == "Tensor" else other),
        )
        return Tensor(
            backend.array(data),
            TensorConfig(
                backend.array(data).shape,
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_linalg_node("Inner", [input, other], {}, [()], [getattr(input, "dtype", None)])


def outer(input: Tensor, other: Tensor) -> Tensor:
    """Compute the outer product of two vectors.

    Args:
        input (Tensor): The input parameter.
        other (Tensor): The other parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "Outer",
            (input.data if type(input).__name__ == "Tensor" else input),
            (other.data if type(other).__name__ == "Tensor" else other),
        )
        return Tensor(
            backend.array(data),
            TensorConfig(
                backend.array(data).shape,
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_linalg_node("Outer", [input, other], {}, [()], [getattr(input, "dtype", None)])


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
        data = backend.execute_op(
            "DotGeneral",
            (lhs.data if type(lhs).__name__ == "Tensor" else lhs),
            (rhs.data if type(rhs).__name__ == "Tensor" else rhs),
            dimension_numbers=dimension_numbers,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, getattr(lhs, "dtype", None), getattr(lhs, "device", None)),
        )
    attributes = {"dimension_numbers": dimension_numbers}
    out_shape = []
    if lhs.shape and rhs.shape:
        out_shape = list(_infer_dot_general_shape(lhs.shape, rhs.shape, dimension_numbers))
    return _emit_linalg_node("DotGeneral", [lhs, rhs], attributes, [tuple(out_shape)], [getattr(lhs, "dtype", None)])


def convolve(a: object, v: object, mode: str = "full") -> Tensor:
    """Return the discrete, linear convolution of two one-dimensional sequences.

    Args:
        a (object): The a parameter.
        v (object): The v parameter.
        mode (str): The mode parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op(
            "Convolve",
            (a.data if type(a).__name__ == "Tensor" else a),
            (v.data if type(v).__name__ == "Tensor" else v),
            mode=mode,
        )
        return Tensor(
            data,
            TensorConfig(data.shape, getattr(a, "dtype", "float32"), getattr(a, "device", None)),
        )
    return _emit_linalg_node("Convolve", [a, v], {"mode": mode}, [(None,)], [getattr(a, "dtype", "float32")])


def matvec(a: object, b: object, transpose_a: object = False, adjoint_a: object = False, **kwargs: object) -> object:
    """Matrix-vector multiplication.

    Args:
        a (object): The a parameter.
        b (object): The b parameter.
        transpose_a (object): The transpose_a parameter.
        adjoint_a (object): The adjoint_a parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return a


def multi_dot(arrays: object, name: object = None) -> object:
    """Evaluate multi_dot operation.

    Args:
        arrays (object): The arrays parameter.
        name (object): The name parameter.

    Returns:
        object: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("MultiDot", [(a.data if type(a).__name__ == "Tensor" else a) for a in arrays])
        return Tensor(data, TensorConfig(data.shape, arrays[0].dtype, arrays[0].device))
    return _emit_linalg_node("MultiDot", arrays, {}, [()], [arrays[0].dtype])


def vecdot(x: object, y: object, axis: object = -1, name: object = None) -> object:
    """Evaluate vecdot operation.

    Args:
        x (object): The x parameter.
        y (object): The y parameter.
        axis (object): The axis parameter.
        name (object): The name parameter.

    Returns:
        object: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "Vecdot",
            (x.data if type(x).__name__ == "Tensor" else x),
            (y.data if type(y).__name__ == "Tensor" else y),
            axis=axis,
        )
        return Tensor(data, TensorConfig(data.shape, getattr(x, "dtype", None), getattr(x, "device", None)))
    return _emit_linalg_node("Vecdot", [x, y], {"axis": axis}, [()], [getattr(x, "dtype", None)])


def addmm(
    input: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    *,
    beta: (float | int) = 1.0,
    alpha: (float | int) = 1.0,
) -> Tensor:
    """Perform a matrix multiplication of the matrices mat1 and mat2.

    Args:
        beta (object): The beta parameter.
        alpha (object): The alpha parameter.
        input (Tensor): The input parameter.
        mat1 (Tensor): The mat1 parameter.
        mat2 (Tensor): The mat2 parameter.

    Returns:
        Tensor: Result.
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
        """Infer shape.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        if isinstance(a, tuple) and isinstance(b, tuple):
            try:
                return matmul_shape(a, b)
            except (ValueError, TypeError, Exception):
                return None
        return getattr(a, "shape", ())


def _unwrap(x: object) -> object:
    """Unwrap a Tensor to its underlying data if it's a Tensor.

    Args:
        x (object): The object to unwrap.

    Returns:
        object: The unwrapped data.
    """
    return x.data if type(x).__name__ == "Tensor" else x


def block_masked_mm(
    a: Tensor,
    b: Tensor,
    block_size: int = 64,
    masks: dict[str, Tensor | None] | None = None,
) -> Tensor:
    """Block masked matrix multiplication.

    Args:
        a (Tensor): The a parameter.
        b (Tensor): The b parameter.
        block_size (int): The block_size parameter.
        masks (object): The masks parameter.

    Returns:
        Tensor: Result.
    """
    masks = masks or {}
    kwargs = {"block_size": block_size}
    if config.eager_mode:
        backend = get_active_backend()
        for k in ("mask_out", "mask_lhs", "mask_rhs"):
            if masks.get(k) is not None:
                kwargs[k] = _unwrap(masks[k])
        data = backend.execute_op("BlockMaskedMm", _unwrap(a), _unwrap(b), **kwargs)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, getattr(a, "dtype", None), getattr(a, "device", None)),
        )
    out_shape = matmul_shape(a.shape, b.shape)
    inputs = [a, b]
    for k in ("mask_out", "mask_lhs", "mask_rhs"):
        if masks.get(k) is not None:
            kwargs[k] = len(inputs)
            inputs.append(masks[k])
    return _emit_shape_node("BlockMaskedMm", inputs, kwargs, out_shape, getattr(a, "dtype", None))


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
        """Infer shape.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            lhs_indices (object): The lhs_indices parameter.
            rhs_indices (object): The rhs_indices parameter.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        if not (isinstance(a, tuple) and isinstance(b, tuple)):
            return getattr(a, "shape", ())
        try:
            mm_shape = matmul_shape(a, b)
        except (ValueError, TypeError, Exception):
            return None
        if isinstance(lhs_indices, tuple):
            return (lhs_indices[0],) + mm_shape[-2:]
        if isinstance(rhs_indices, tuple):
            return (rhs_indices[0],) + mm_shape[-2:]
        return mm_shape


def _gather_mm_infer_shape(a: Tensor, b: Tensor, lhs_indices: Tensor = None, rhs_indices: Tensor = None) -> tuple:
    """Infer shape for gather_mm operation.

    Args:
        a (Tensor): The a parameter.
        b (Tensor): The b parameter.
        lhs_indices (Tensor): The lhs_indices parameter.
        rhs_indices (Tensor): The rhs_indices parameter.

    Returns:
        tuple: Result.
    """
    out_shape = list(getattr(a, "shape", (1, 1, 1)))
    if lhs_indices is not None:
        out_shape[0] = getattr(lhs_indices, "shape", (1,))[0]
    elif rhs_indices is not None:
        out_shape[0] = getattr(rhs_indices, "shape", (1,))[0]
    return tuple(out_shape[:-2] + [getattr(a, "shape", (1, 1))[-2], getattr(b, "shape", (1, 1))[-1]])


def gather_mm(
    a: Tensor,
    b: Tensor,
    lhs_indices: Tensor = None,
    rhs_indices: Tensor = None,
    sorted_indices: bool = False,
) -> Tensor:
    """Gather matrix multiplication.

    Args:
        a (Tensor): The a parameter.
        b (Tensor): The b parameter.
        lhs_indices (Tensor): The lhs_indices parameter.
        rhs_indices (Tensor): The rhs_indices parameter.
        sorted_indices (bool): The sorted_indices parameter.

    Returns:
        Tensor: Result.
    """
    kwargs = {"sorted_indices": sorted_indices}
    if config.eager_mode:
        backend = get_active_backend()
        for k, v in (("lhs_indices", lhs_indices), ("rhs_indices", rhs_indices)):
            if v is not None:
                kwargs[k] = _unwrap(v)
        data = backend.execute_op("GatherMm", _unwrap(a), _unwrap(b), **kwargs)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, getattr(a, "dtype", None), getattr(a, "device", None)),
        )
    out_shape = _gather_mm_infer_shape(a, b, lhs_indices, rhs_indices)
    inputs = [a, b]
    for k, v in (("lhs_indices", lhs_indices), ("rhs_indices", rhs_indices)):
        if v is not None:
            kwargs[k] = len(inputs)
            inputs.append(v)
    return _emit_shape_node("GatherMm", inputs, kwargs, out_shape, getattr(a, "dtype", None))


@register_op("SegmentedMm")
class SegmentedMm(OpDef):
    """SegmentedMm operation."""

    op_name = "SegmentedMm"

    def infer_shape(self, a: object, b: object, segments: object = None, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            segments (object): The segments parameter.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        if isinstance(a, tuple) and isinstance(b, tuple):
            if segments is not None and isinstance(segments, tuple):
                return segments[0] - 1, a[-2], b[-1]
            return 1, a[-2], b[-1]
        return getattr(a, "shape", ())


def segmented_mm(a: Tensor, b: Tensor, segments: Tensor) -> Tensor:
    """Segmented matrix multiplication.

    Args:
        a (Tensor): The a parameter.
        b (Tensor): The b parameter.
        segments (Tensor): The segments parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "SegmentedMm",
            (a.data if type(a).__name__ == "Tensor" else a),
            (b.data if type(b).__name__ == "Tensor" else b),
            segments=(segments.data if type(segments).__name__ == "Tensor" else segments),
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, getattr(a, "dtype", None), getattr(a, "device", None)),
        )
    out_shape = (
        getattr(segments, "shape", (2,))[0] - 1,
        getattr(a, "shape", (1, 1))[-2],
        getattr(b, "shape", (1, 1))[-1],
    )
    return _emit_shape_node("SegmentedMm", [a, b, segments], {"segments": 2}, out_shape, getattr(a, "dtype", None))
