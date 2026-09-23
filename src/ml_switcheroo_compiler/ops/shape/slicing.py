"""Module slicing.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Shape operations for Tensor Anys."""
import builtins
import math
from collections.abc import Sequence

# pylint: disable=duplicate-code
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def slice(
    input: Tensor,
    axis: int,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
):
    """Slice the input tensor along a specific dimension.

    Args:
        input (Tensor): The input parameter.
        axis (int): The axis parameter.
        start (Any): The start parameter.
        end (Any): The end parameter.
        step (int): The step parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        sl = [builtins.slice(None)] * len(input.shape)
        sl[axis] = builtins.slice(start, end, step)
        data = input.data[tuple(sl)]
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

        en = end if end is not None else (dim_len if s > 0 else -1)
        if en < 0 and end is not None:
            en = max(0, en + dim_len)
        elif end is not None:
            en = min(dim_len, en)

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
):
    """Extract a strided slice from the input tensor.

    Args:
        input (Tensor): The input parameter.
        begin (Sequence): The begin parameter.
        end (Sequence): The end parameter.
        strides (Sequence): The strides parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        idx = tuple(builtins.slice(b, e, s) for b, e, s in zip(begin, end, strides))
        data = input.data[idx]
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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("StridedSlice")
class StridedSlice(OpDef):
    """StridedSlice operator definition."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape for StridedSlice.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple: Result.
        """
        return ()


@register_op("Choose")
class Choose(OpDef):
    """Construct an array from an index array and a list of arrays to choose from."""

    op_name = "Choose"
    np_op_name = "choose"

    def infer_shape(self, a, choices, out=None, mode: str = "raise", **kwargs):
        """Infer the output shape.

        Args:
            a (Any): The a parameter.
            choices (Any): The choices parameter.
            out (Any): The out parameter.
            mode (str): The mode parameter.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return a.shape if hasattr(a, "shape") else ()


@register_op("IndexInDim")
class IndexInDim(OpDef):
    """Return elements of an array at specific indices along a given dimension."""

    op_name = "IndexInDim"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        operand = args[0] if len(args) > 0 else None
        index = args[1] if len(args) > 1 else None
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", True)
        shape = list(getattr(operand, "shape", ()))
        if not shape:
            return ()

        index_shape = getattr(index, "shape", ())
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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        operand = args[0] if len(args) > 0 else None
        return getattr(operand, "shape", ())


def index_in_dim(*args, **kwargs):
    """Return the index in a dimension.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("IndexInDim", *args, **kwargs)


def update_slice(*args, **kwargs):
    """Update a slice.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("UpdateSlice", *args, **kwargs)
