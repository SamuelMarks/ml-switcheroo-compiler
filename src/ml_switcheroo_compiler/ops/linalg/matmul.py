"""Implementation of matmul operations and utilities for the ops subsystem (`ml_switcheroo_compiler.ops.linalg.matmul`)."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import (
    ensure_callable,
    ensure_eager_list,
    ensure_eager_tensor,
    get_active_backend,
)
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import (
    TensorOrScalar,
    extract_data,
    extract_device,
    extract_dtype,
    extract_shape,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig, as_opt_tensor_obj
from ml_switcheroo_compiler.ir.shape_system import matmul_shape
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.binary import add, multiply
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

from .einsum_frontend import _infer_dot_general_shape
from .utils import _emit_linalg_node, _extract_linalg_dtype


def matmul(input: Tensor, other: Tensor) -> TensorOrScalar:
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
            TensorConfig(extract_shape(backend.array(data)), _extract_linalg_dtype(input, None), extract_device(input, None)),
        )
    out_shape = matmul_shape(input.shape, other.shape)
    return _emit_linalg_node("Matmul", [input, other], {}, [out_shape], [_extract_linalg_dtype(input, None)])


def dot(input: Tensor, other: Tensor) -> TensorOrScalar:
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
                extract_shape(backend.array(data)),
                extract_dtype(input, "float32"),
                extract_device(input, None),
            ),
        )
    return _emit_linalg_node("Dot", [input, other], {}, [()], [_extract_linalg_dtype(input, None)])


def vdot(input: Tensor, other: Tensor) -> TensorOrScalar:
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
                extract_shape(backend.array(data)),
                extract_dtype(input, "float32"),
                extract_device(input, None),
            ),
        )
    return _emit_linalg_node("Vdot", [input, other], {}, [()], [_extract_linalg_dtype(input, None)])


def inner(input: Tensor, other: Tensor) -> TensorOrScalar:
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
                extract_shape(backend.array(data)),
                extract_dtype(input, "float32"),
                extract_device(input, None),
            ),
        )
    return _emit_linalg_node("Inner", [input, other], {}, [()], [_extract_linalg_dtype(input, None)])


def outer(input: Tensor, other: Tensor) -> TensorOrScalar:
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
                extract_shape(backend.array(data)),
                extract_dtype(input, "float32"),
                extract_device(input, None),
            ),
        )
    return _emit_linalg_node("Outer", [input, other], {}, [()], [_extract_linalg_dtype(input, None)])


def dot_general(
    lhs: Tensor,
    rhs: Tensor,
    dimension_numbers: tuple[tuple[Sequence[int], Sequence[int]], tuple[Sequence[int], Sequence[int]]],
) -> TensorOrScalar:
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
            TensorConfig(extract_shape(backend.array(data)), _extract_linalg_dtype(lhs, None), extract_device(lhs, None)),
        )
    attributes = {"dimension_numbers": dimension_numbers}
    out_shape = []
    if lhs.shape and rhs.shape:
        out_shape = list(_infer_dot_general_shape(lhs.shape, rhs.shape, dimension_numbers))
    return ensure_eager_tensor(ensure_callable(_emit_linalg_node)("DotGeneral", [lhs, rhs], attributes, [tuple(out_shape)], [_extract_linalg_dtype(lhs, None)]))


def convolve(a: TensorOrScalar, v: TensorOrScalar, mode: str = "full") -> TensorOrScalar:
    """Return the discrete, linear convolution of two one-dimensional sequences.

    Args:
        a (TensorOrScalar): The a parameter.
        v (TensorOrScalar): The v parameter.
        mode (str): The mode parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op(
            "Convolve",
            (extract_data(a) if type(a).__name__ == "Tensor" else a),
            (extract_data(v) if type(v).__name__ == "Tensor" else v),
            mode=mode,
        )
        return Tensor(
            data,
            TensorConfig(extract_shape(data), extract_dtype(a, "float32"), extract_device(a, None)),
        )
    return _emit_linalg_node("Convolve", [a, v], {"mode": mode}, [(None,)], [extract_dtype(a, "float32")])


def matvec(a: TensorOrScalar, b: TensorOrScalar, transpose_a: TensorOrScalar = False, adjoint_a: TensorOrScalar = False, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Matrix-vector multiplication.

    Args:
        a (TensorOrScalar): The a parameter.
        b (TensorOrScalar): The b parameter.
        transpose_a (TensorOrScalar): The transpose_a parameter.
        adjoint_a (TensorOrScalar): The adjoint_a parameter.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return a


def multi_dot(arrays: TensorOrScalar, name: TensorOrScalar = None) -> TensorOrScalar:
    """Compute the multi dot tensor operation.

    Args:
        arrays (TensorOrScalar): The arrays parameter.
        name (TensorOrScalar): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    arr_list = ensure_eager_list(arrays)
    first_arr = arr_list[0]
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("MultiDot", [(extract_data(a) if type(a).__name__ == "Tensor" else a) for a in arr_list])
        return Tensor(data, TensorConfig(extract_shape(data), _extract_linalg_dtype(first_arr, None), extract_device(first_arr, None)))
    return ensure_eager_tensor(ensure_callable(_emit_linalg_node)("MultiDot", arrays, {}, [()], [_extract_linalg_dtype(first_arr, None)]))


def vecdot(x: TensorOrScalar, y: TensorOrScalar, axis: TensorOrScalar = -1, name: TensorOrScalar = None) -> TensorOrScalar:
    """Compute the vecdot tensor operation.

    Args:
        x (TensorOrScalar): The x parameter.
        y (TensorOrScalar): The y parameter.
        axis (TensorOrScalar): The axis parameter.
        name (TensorOrScalar): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "Vecdot",
            (extract_data(x) if type(x).__name__ == "Tensor" else x),
            (extract_data(y) if type(y).__name__ == "Tensor" else y),
            axis=axis,
        )
        return Tensor(data, TensorConfig(extract_shape(data), _extract_linalg_dtype(x, None), extract_device(x, None)))
    return _emit_linalg_node("Vecdot", [x, y], {"axis": axis}, [()], [_extract_linalg_dtype(x, None)])


def addmm(
    input: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    *,
    beta: (float | int) = 1.0,
    alpha: (float | int) = 1.0,
) -> TensorOrScalar:
    """Perform a matrix multiplication of the matrices mat1 and mat2.

    Args:
        beta (float | int): The beta parameter.
        alpha (float | int): The alpha parameter.
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

    def infer_shape(self, a: TensorOrScalar, b: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            a (EagerTensorProtocol): The a parameter.
            b (EagerTensorProtocol): The b parameter.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        if isinstance(a, tuple) and isinstance(b, tuple):
            try:
                return matmul_shape(a, b)
            except (ValueError, TypeError, Exception):
                return None
        _typed_ret_1: TensorOrScalar = extract_shape(a)
        return _typed_ret_1


def _unwrap(x: Tensor | None) -> TensorOrScalar | None:
    """Unwrap a Tensor to its underlying data if it's a Tensor.

    Args:
        x (Tensor | None): The object to unwrap.

    Returns: Tensor: The unwrapped data.
    """
    return extract_data(x) if type(x).__name__ == "Tensor" else x


def block_masked_mm(
    a: Tensor,
    b: Tensor,
    block_size: int = 64,
    masks: dict[str, Tensor | None] | None = None,
) -> TensorOrScalar:
    """Block masked matrix multiplication.

    Args:
        a (Tensor): The a parameter.
        b (Tensor): The b parameter.
        block_size (int): The block_size parameter.
        masks (dict[str, Tensor | None] | None): The masks parameter.

    Returns:
        Tensor: Result.
    """
    masks = masks or {}

    kwargs: dict[str, TensorOrScalar] = {"block_size": block_size}
    if config.eager_mode:
        backend = get_active_backend()
        for k in ("mask_out", "mask_lhs", "mask_rhs"):
            if masks.get(k) is not None:
                kwargs[k] = _unwrap(masks[k])
        data = backend.execute_op("BlockMaskedMm", _unwrap(a), _unwrap(b), **kwargs)
        return Tensor(
            backend.array(data),
            TensorConfig(extract_shape(backend.array(data)), _extract_linalg_dtype(a, None), extract_device(a, None)),
        )
    out_shape = matmul_shape(a.shape, b.shape)
    inputs: list[TensorOrScalar] = [a, b]
    for k in ("mask_out", "mask_lhs", "mask_rhs"):
        if masks.get(k) is not None:
            kwargs[k] = len(inputs)
            inputs.append(masks[k])
    return _emit_shape_node("BlockMaskedMm", inputs, kwargs, out_shape, _extract_linalg_dtype(a, None))


@register_op("GatherMm")
class GatherMm(OpDef):
    """GatherMm operation."""

    op_name = "GatherMm"

    def infer_shape(
        self,
        a: TensorOrScalar,
        b: TensorOrScalar,
        lhs_indices: TensorOrScalar = None,
        rhs_indices: TensorOrScalar = None,
        **kwargs: TensorOrScalar,
    ) -> TensorOrScalar:
        """Infer shape.

        Args:
            a (EagerTensorProtocol): The a parameter.
            b (EagerTensorProtocol): The b parameter.
            lhs_indices (TensorOrScalar): The lhs_indices parameter.
            rhs_indices (TensorOrScalar): The rhs_indices parameter.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        if not (isinstance(a, tuple) and isinstance(b, tuple)):
            _typed_ret_3: TensorOrScalar = extract_shape(a)
            return _typed_ret_3
        try:
            mm_shape = matmul_shape(a, b)
        except (ValueError, TypeError, Exception):
            return None
        if isinstance(lhs_indices, tuple):
            return (lhs_indices[0],) + mm_shape[-2:]
        if isinstance(rhs_indices, tuple):
            return (rhs_indices[0],) + mm_shape[-2:]
        return mm_shape


def _gather_mm_infer_shape(a: Tensor, b: Tensor, lhs_indices: TensorOrScalar = None, rhs_indices: TensorOrScalar = None) -> TensorOrScalar:
    """Infer shape for gather_mm operation.

    Args:
        a (Tensor): The a parameter.
        b (Tensor): The b parameter.
        lhs_indices (Tensor): The lhs_indices parameter.
        rhs_indices (Tensor): The rhs_indices parameter.

    Returns:
        tuple: Result.
    """
    a_shape = extract_shape(a, (1, 1, 1))
    b_shape = extract_shape(b, (1, 1))
    out_shape = list(a_shape)
    if lhs_indices is not None:
        out_shape[0] = extract_shape(lhs_indices, (1,))[0]
    elif rhs_indices is not None:
        out_shape[0] = extract_shape(rhs_indices, (1,))[0]
    return tuple(out_shape[:-2] + [a_shape[-2], b_shape[-1]])


def gather_mm(
    a: Tensor,
    b: Tensor,
    lhs_indices: TensorOrScalar = None,
    rhs_indices: TensorOrScalar = None,
    sorted_indices: bool = False,
) -> TensorOrScalar:
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
    kwargs: dict[str, TensorOrScalar] = {"sorted_indices": sorted_indices}
    if config.eager_mode:
        backend = get_active_backend()
        for k, v in (("lhs_indices", lhs_indices), ("rhs_indices", rhs_indices)):
            if v is not None:
                kwargs[k] = _unwrap(as_opt_tensor_obj(v))
        data = backend.execute_op("GatherMm", _unwrap(a), _unwrap(b), **kwargs)
        return Tensor(
            backend.array(data),
            TensorConfig(extract_shape(backend.array(data)), _extract_linalg_dtype(a, None), extract_device(a, None)),
        )
    out_shape = _gather_mm_infer_shape(a, b, lhs_indices, rhs_indices)
    inputs: list[TensorOrScalar] = [a, b]
    for k, v in (("lhs_indices", lhs_indices), ("rhs_indices", rhs_indices)):
        if v is not None:
            kwargs[k] = len(inputs)
            inputs.append(v)
    return _emit_shape_node("GatherMm", inputs, kwargs, out_shape, _extract_linalg_dtype(a, None))


@register_op("SegmentedMm")
class SegmentedMm(OpDef):
    """SegmentedMm operation."""

    op_name = "SegmentedMm"

    def infer_shape(self, a: TensorOrScalar, b: TensorOrScalar, segments: TensorOrScalar = None, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            a (EagerTensorProtocol): The a parameter.
            b (EagerTensorProtocol): The b parameter.
            segments (TensorOrScalar): The segments parameter.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        if isinstance(a, tuple) and isinstance(b, tuple):
            if segments is not None and isinstance(segments, tuple):
                return segments[0] - 1, a[-2], b[-1]
            return 1, a[-2], b[-1]
        _typed_ret_2: TensorOrScalar = extract_shape(a)
        return _typed_ret_2


def segmented_mm(a: Tensor, b: Tensor, segments: Tensor) -> TensorOrScalar:
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
            TensorConfig(extract_shape(backend.array(data)), _extract_linalg_dtype(a, None), extract_device(a, None)),
        )
    out_shape = (
        extract_shape(segments, (2,))[0] - 1,
        extract_shape(a, (1, 1))[-2],
        extract_shape(b, (1, 1))[-1],
    )
    return _emit_shape_node("SegmentedMm", [a, b, segments], {"segments": 2}, out_shape, _extract_linalg_dtype(a, None))
