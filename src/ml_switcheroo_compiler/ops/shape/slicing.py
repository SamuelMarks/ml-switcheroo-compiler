"""Shape operations for Tensor objects."""

from __future__ import annotations

import builtins
import math
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, ensure_eager_tensor, ensure_int

# pylint: disable=duplicate-code
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar, extract_shape
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node, _extract_shape_or_seq


def slice(
    input: Tensor,
    axis: int,
    start: int | None = None,
    end: int | None = None,
    step: int | None = 1,
) -> TensorOrScalar:
    """Slice the input tensor along a specific dimension.

    Args:
        input (Tensor): Input `input` argument (input).
        axis (int): Input `axis` argument (axis).
        start (int | None): Input `start` argument (start).
        end (int | None): Input `end` argument (end).
        step (int): Input `step` argument (step).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        sl = [builtins.slice(None)] * len(input.shape)
        sl[axis] = builtins.slice(start, end, step)
        data = ensure_eager_tensor(input.data)[tuple(sl)]
        return Tensor(data, TensorConfig(data.shape, input.dtype, input.device))
    inputs = [input]
    base_shape = inputs[0].shape
    rank = len(base_shape)
    norm_axis = axis if axis >= 0 else axis + rank
    if rank > 0 and 0 <= norm_axis < rank:
        dim_len = int(base_shape[norm_axis])
        s = step if step is not None else 1
        st = start if start is not None else (0 if s > 0 else dim_len - 1)
        if st < 0:
            st = max(0, st + dim_len)
        else:
            st = min(dim_len, st)

        if end is None:
            en = dim_len if s > 0 else -1
        elif end < 0:
            en = max(0, end + dim_len)
        else:
            en = min(dim_len, end)

        if s > 0:
            slice_dim = max(0, math.ceil((en - st) / s))
        else:
            slice_dim = max(0, math.ceil((st - en) / (-s)))
        out_shape = tuple(slice_dim if i == norm_axis else d for i, d in enumerate(base_shape))
    else:
        out_shape = base_shape
    return _emit_shape_node(
        "Slice",
        inputs,
        {"axis": axis, "start": start, "end": end, "step": step},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def strided_slice(
    input: Tensor,
    begin: Sequence[int],
    end: Sequence[int],
    strides: Sequence[int],
) -> TensorOrScalar:
    """Extract a strided slice from the input tensor.

    Args:
        input (Tensor): Input `input` argument (input).
        begin (Sequence): Input `begin` argument (begin).
        end (Sequence): Input `end` argument (end).
        strides (Sequence): Input `strides` argument (strides).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        idx = tuple(builtins.slice(b, e, s) for b, e, s in zip(begin, end, strides))
        data = ensure_eager_tensor(input.data)[idx]
        return Tensor(data, TensorConfig(data.shape, input.dtype, input.device))
    inputs = [input]
    base_shape = inputs[0].shape
    out_dims = []
    for i, d in enumerate(base_shape):
        b = begin[i] if i < len(begin) else 0
        e = end[i] if i < len(end) else d
        s = strides[i] if i < len(strides) else 1
        dim_val = int(d)
        st = b if b >= 0 else max(0, b + dim_val)
        st = min(dim_val, st)
        en = e if e >= 0 else max(0, e + dim_val)
        en = min(dim_val, en)
        if s > 0:
            out_dims.append(max(0, math.ceil((en - st) / s)))
        else:
            out_dims.append(max(0, math.ceil((st - en) / (-s))))
    out_shape = tuple(out_dims)
    return _emit_shape_node(
        "StridedSlice",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


@register_op("Slice")
class Slice(OpDef):
    """Slice operator definition."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return ()


@register_op("StridedSlice")
class StridedSlice(OpDef):
    """StridedSlice operator definition."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape for StridedSlice.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple: Computed `tuple` output.
        """
        return ()


@register_op("Choose")
class Choose(OpDef):
    """Construct an array from an index array and a list of arrays to choose from."""

    op_name = "Choose"
    np_op_name = "choose"

    def infer_shape(self, a: EagerTensorProtocol, choices: EagerTensorProtocol, out: TensorOrScalar = None, mode: str = "raise", **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the output shape.

        Args:
            a (EagerTensorProtocol): Input `a` argument (a).
            choices (EagerTensorProtocol): Input `choices` argument (choices).
            out (TensorOrScalar): Input `out` argument (out).
            mode (str): Input `mode` argument (mode).
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return extract_shape(a)


@register_op("IndexInDim")
class IndexInDim(OpDef):
    """Return elements of an array at specific indices along a given dimension."""

    op_name = "IndexInDim"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer shape.

        Args:
            *args (EagerTensorProtocol): Positional args.
            **kwargs (EagerTensorProtocol): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        operand = args[0] if len(args) > 0 else kwargs.get("operand", kwargs.get("a"))
        index = args[1] if len(args) > 1 else kwargs.get("index", kwargs.get("indices"))
        axis = int(ensure_int(kwargs.get("axis", 0)))
        keepdims = bool(kwargs.get("keepdims", True))
        shape = list(_extract_shape_or_seq(operand))
        if not shape:
            return ()

        if axis < 0:
            axis += len(shape)

        index_shape = _extract_shape_or_seq(index)
        if keepdims:
            shape[axis] = index_shape[0] if index_shape else 1
        else:
            if index_shape:
                shape.pop(axis)
                shape = shape[:axis] + list(index_shape) + shape[axis:]
            else:
                shape.pop(axis)

        return tuple(shape)


@register_op("UpdateSlice")
class UpdateSlice(OpDef):
    """Update a slice of an array."""

    op_name = "UpdateSlice"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer shape.

        Args:
            *args (EagerTensorProtocol): Positional args.
            **kwargs (EagerTensorProtocol): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        operand = args[0] if len(args) > 0 else kwargs.get("operand", kwargs.get("a"))
        if operand is None:
            return ()
        return _extract_shape_or_seq(operand)


def index_in_dim(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Return the index in a dimension.

    Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("IndexInDim", *args, **kwargs)


def update_slice(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Update a slice.

    Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("UpdateSlice", *args, **kwargs)
