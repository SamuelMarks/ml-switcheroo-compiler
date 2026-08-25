"""Module slicing.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Shape operations for Tensor objects."""
import builtins
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
) -> object:
    """Slice the input tensor along a specific dimension.

    Args:
        input (Tensor): The input parameter.
        axis (int): The axis parameter.
        start (object): The start parameter.
        end (object): The end parameter.
        step (int): The step parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        sl: object = [builtins.slice(None)] * len(input.shape)
        sl[axis] = builtins.slice(start, end, step)
        data: object = input.data[tuple(sl)]
        return Tensor(data, TensorConfig(data.shape, input.dtype, input.device))
    inputs: object = [input]
    # shape calculation placeholder
    out_shape: object = inputs[0].shape
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
) -> object:
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
        idx: object = tuple(builtins.slice(b, e, s) for b, e, s in zip(begin, end, strides))
        data: object = input.data[idx]
        return Tensor(data, TensorConfig(data.shape, input.dtype, input.device))
    inputs: object = [input]
    # shape calculation placeholder
    out_shape: object = inputs[0].shape
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

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("StridedSlice")
class StridedSlice(OpDef):
    """StridedSlice operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape for StridedSlice.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple: Result.
        """
        return ()


@register_op("Choose")
class Choose(OpDef):
    """Construct an array from an index array and a list of arrays to choose from."""

    op_name: object = "Choose"
    np_op_name: object = "choose"

    def infer_shape(self, a: object, choices: object, out: object = None, mode: str = "raise", **kwargs: object) -> object:
        """Infer the output shape.

        Args:
            a (object): The a parameter.
            choices (object): The choices parameter.
            out (object): The out parameter.
            mode (str): The mode parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return a.shape if hasattr(a, "shape") else ()


@register_op("IndexInDim")
class IndexInDim(OpDef):
    """Return elements of an array at specific indices along a given dimension."""

    op_name: object = "IndexInDim"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        operand: object = args[0] if len(args) > 0 else None
        index: object = args[1] if len(args) > 1 else None
        axis: object = kwargs.get("axis", 0)
        keepdims: object = kwargs.get("keepdims", True)
        shape: object = list(getattr(operand, "shape", ()))
        if not shape:
            return ()

        index_shape: object = getattr(index, "shape", ())
        if keepdims:
            shape[axis] = index_shape[0] if index_shape else 1
        else:
            if index_shape:
                shape.pop(axis)
                shape: object = shape[:axis] + list(index_shape) + shape[axis:]
            else:
                shape.pop(axis)

        return tuple(shape)


@register_op("UpdateSlice")
class UpdateSlice(OpDef):
    """Update a slice of an array."""

    op_name: object = "UpdateSlice"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        operand: object = args[0] if len(args) > 0 else None
        return getattr(operand, "shape", ())


def index_in_dim(*args: object, **kwargs: object) -> object:
    """Return the index in a dimension.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("IndexInDim", *args, **kwargs)


def update_slice(*args: object, **kwargs: object) -> object:
    """Update a slice.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("UpdateSlice", *args, **kwargs)
