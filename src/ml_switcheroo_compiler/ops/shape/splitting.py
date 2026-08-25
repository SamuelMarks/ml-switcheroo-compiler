"""Module splitting.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Shape operations for Tensor objects."""
from collections.abc import Sequence

# pylint: disable=duplicate-code
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node
from ml_switcheroo_compiler.tracing import builder


def _calculate_num_splits(input: object, split_size_or_sections: int | Sequence[int], axis: int) -> object:  #
    """Calculate the number of splits for a tensor.

    Args:
        input (Tensor): The input tensor
        split_size_or_sections (int | Sequence[int]): The size of a single chunk or list of sizes
        axis (int): The dimension along which to split

    Returns:
        int: The number of splits
    """
    if not isinstance(split_size_or_sections, int):
        return len(split_size_or_sections) + 1

    if getattr(input, "shape", None) is None:
        return split_size_or_sections

    if split_size_or_sections <= 0:
        return 1

    if input.shape and input.shape[axis] % split_size_or_sections == 0:
        return input.shape[axis] // split_size_or_sections

    return split_size_or_sections


def _validate_split_axis(input: object, axis: int) -> object:  #
    """Validate the split axis against the input tensor shape.

    Args:
        input (Tensor): The input parameter.
        axis (int): The axis parameter.

    Returns:
        int: Result.

    Raises:
        ValueError: An exception.
    """
    shape: object = getattr(input, "shape", None)
    if shape is None:
        return axis
    rank: object = len(shape)
    if not (0 <= axis < rank or -rank <= axis < 0):
        raise ValueError(f"Split axis {axis} is out of bounds for tensor of rank {rank}.")
    return axis


# pylint: disable=too-many-locals


def _split_even(input: object, split_size: int, axis: int, num_splits: int) -> Sequence[object]:
    """Handle splitting when an integer split_size_or_sections is provided.

    Args:
        input (Tensor): The input parameter.
        split_size (int): The split_size parameter.
        axis (int): The axis parameter.
        num_splits (int): The num_splits parameter.

    Returns:
        Sequence: Result.
    """
    if config.eager_mode:
        backend: object = get_active_backend()
        input_data: object = getattr(input, "data", input)
        datas: object = backend.execute_op("Split", input_data, split_size, axis=axis)
        input_dtype: object = getattr(input, "dtype", backend.array(input_data).dtype)
        input_device: object = getattr(input, "device", config.default_device)
        return tuple(Tensor(d, TensorConfig(d.shape, input_dtype, input_device)) for d in datas)

    inputs: object = [input]
    out_shape: object = inputs[0].shape if len(inputs) > 0 else ()
    node: object = _emit_shape_node(
        "Split",
        inputs,
        {"split_size_or_sections": split_size, "axis": axis},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )

    out_tensors: object = []
    for i in range(num_splits):
        item_node: object = builder.TracingNodeBuilder.emit_tracing_node("GetItem", node, output_index=i, key=str(i))
        out_tensors.append(item_node)
    return tuple(out_tensors)


def _split_sections(  # pylint: disable=too-many-locals
    input: object,
    sections: Sequence[int],
    axis: int,
    num_splits: int,  #
) -> Sequence[object]:
    """Handle splitting when a sequence of sections is provided.

    Args:
        input (Tensor): The input parameter.
        sections (Sequence): The sections parameter.
        axis (int): The axis parameter.
        num_splits (int): The num_splits parameter.

    Returns:
        Sequence: Result.
    """
    if config.eager_mode:
        backend: object = get_active_backend()
        input_data: object = getattr(input, "data", input)
        datas: object = backend.execute_op("Split", input_data, sections, axis=axis)
        input_dtype: object = getattr(input, "dtype", backend.array(input_data).dtype)
        input_device: object = getattr(input, "device", config.default_device)
        return tuple(Tensor(d, TensorConfig(d.shape, input_dtype, input_device)) for d in datas)

    inputs: object = [input]
    out_shape: object = inputs[0].shape if len(inputs) > 0 else ()
    node: object = _emit_shape_node(
        "Split",
        inputs,
        {"split_size_or_sections": sections, "axis": axis},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )

    out_tensors: object = []
    for i in range(num_splits):
        item_node: object = builder.TracingNodeBuilder.emit_tracing_node("GetItem", node, output_index=i, key=str(i))
        out_tensors.append(item_node)
    return tuple(out_tensors)


def split(
    input: object,  #
    split_size_or_sections: int | Sequence[int],
    axis: int = 0,
) -> Sequence[object]:
    """Split the input tensor into multiple sub-tensors.

    Args:
        input (Tensor): The input parameter.
        split_size_or_sections (object): The split_size_or_sections parameter.
        axis (int): The axis parameter.

    Returns:
        Sequence: Result.
    """
    valid_axis: object = _validate_split_axis(input, axis)
    num_splits: object = _calculate_num_splits(input, split_size_or_sections, valid_axis)

    if isinstance(split_size_or_sections, int):
        return _split_even(input, split_size_or_sections, valid_axis, num_splits)
    return _split_sections(input, split_size_or_sections, valid_axis, num_splits)


def unstack(input: object, axis: int = 0) -> Sequence[object]:
    """Unstack the input tensor along a specified dimension into a sequence of tensors.

    Args:
        input (Tensor): The input parameter.
        axis (int): The axis parameter.

    Returns:
        Sequence: Result.
    """
    if config.eager_mode:
        backend: object = get_active_backend()
        datas: object = backend.execute_op("Unstack", input.data, axis=axis) if hasattr(backend, "unstack") else backend.execute_op("Moveaxis", input.data, axis, 0)
        input_data: object = getattr(input, "data", input)
        input_dtype: object = getattr(input, "dtype", backend.array(input_data).dtype)
        input_device: object = getattr(input, "device", config.default_device)
        return tuple(Tensor(d, TensorConfig(d.shape, input_dtype, input_device)) for d in datas)
    inputs: object = [input]
    # shape calculation placeholder
    out_shape: object = inputs[0].shape if len(inputs) > 0 else ()
    return (
        _emit_shape_node(
            "Unstack",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        ),
    )


def array_split(
    ary: object,  #
    indices_or_sections: int | Sequence[int],
    axis: int = 0,
) -> Sequence[object]:
    """Split an array into multiple sub-arrays.

    Args:
        ary (Tensor): The ary parameter.
        indices_or_sections (object): The indices_or_sections parameter.
        axis (int): The axis parameter.

    Returns:
        Sequence: Result.
    """
    if config.eager_mode:
        backend: object = get_active_backend()
        datas: object = backend.execute_op("ArraySplit", ary.data, indices_or_sections, axis=axis)
        return tuple(Tensor(d, TensorConfig(d.shape, ary.dtype, ary.device)) for d in datas)
    return (
        _emit_shape_node(
            "ArraySplit",
            [ary],
            {"indices_or_sections": indices_or_sections, "axis": axis},
            ary.shape,
            ary.dtype,
        ),
    )


def vsplit(ary: object, indices_or_sections: int | Sequence[int]) -> Sequence[object]:
    """Split an array into multiple sub-arrays vertically (row-wise).

    Args:
        ary (Tensor): The ary parameter.
        indices_or_sections (object): The indices_or_sections parameter.

    Returns:
        Sequence: Result.
    """
    if config.eager_mode:
        backend: object = get_active_backend()
        datas: object = backend.execute_op("Vsplit", ary.data, indices_or_sections)
        return tuple(Tensor(d, TensorConfig(d.shape, ary.dtype, ary.device)) for d in datas)

    num_splits: object = len(indices_or_sections) + 1 if not isinstance(indices_or_sections, int) else indices_or_sections
    node: object = _emit_shape_node("Vsplit", [ary], {"indices_or_sections": indices_or_sections}, ary.shape, ary.dtype)
    out_tensors: object = []

    # Calculate output shapes
    out_shapes: object = []
    if isinstance(indices_or_sections, int):
        s: object = list(ary.shape)
        if len(s) > 0:
            s[0] = s[0] // indices_or_sections
        out_shapes: object = [tuple(s)] * num_splits
    else:
        # Just approximate
        out_shapes: object = [ary.shape] * num_splits  #   # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    for i in range(num_splits):
        item_node: object = builder.TracingNodeBuilder.emit_tracing_node("GetItem", node, output_index=i, key=str(i))
        item_node._shape = out_shapes[i]
        item_node.config = TensorConfig(out_shapes[i], item_node.dtype, item_node.device)
        out_tensors.append(item_node)
    return tuple(out_tensors)


def hsplit(ary: object, indices_or_sections: int | Sequence[int]) -> Sequence[object]:
    """Split an array into multiple sub-arrays horizontally (column-wise).

    Args:
        ary (Tensor): The ary parameter.
        indices_or_sections (object): The indices_or_sections parameter.

    Returns:
        Sequence: Result.
    """
    if config.eager_mode:
        backend: object = get_active_backend()
        datas: object = backend.execute_op("Hsplit", ary.data, indices_or_sections)
        return tuple(Tensor(d, TensorConfig(d.shape, ary.dtype, ary.device)) for d in datas)

    num_splits: object = len(indices_or_sections) + 1 if not isinstance(indices_or_sections, int) else indices_or_sections
    node: object = _emit_shape_node("Hsplit", [ary], {"indices_or_sections": indices_or_sections}, ary.shape, ary.dtype)
    out_tensors: object = []

    out_shapes: object = []
    if isinstance(indices_or_sections, int):
        s: object = list(ary.shape)
        if len(s) > 1:
            s[1] = s[1] // indices_or_sections
        out_shapes: object = [tuple(s)] * num_splits
    else:
        out_shapes: object = [ary.shape] * num_splits  #   # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    for i in range(num_splits):
        item_node: object = builder.TracingNodeBuilder.emit_tracing_node("GetItem", node, output_index=i, key=str(i))
        item_node._shape = out_shapes[i]
        item_node.config = TensorConfig(out_shapes[i], item_node.dtype, item_node.device)
        out_tensors.append(item_node)
    return tuple(out_tensors)


def dsplit(ary: object, indices_or_sections: int | Sequence[int]) -> Sequence[object]:
    """Split array into multiple sub-arrays along the 3rd axis (depth).

    Args:
        ary (Tensor): The ary parameter.
        indices_or_sections (object): The indices_or_sections parameter.

    Returns:
        Sequence: Result.
    """
    if config.eager_mode:
        backend: object = get_active_backend()
        datas: object = backend.execute_op("Dsplit", ary.data, indices_or_sections)
        return tuple(Tensor(d, TensorConfig(d.shape, ary.dtype, ary.device)) for d in datas)

    num_splits: object = len(indices_or_sections) + 1 if not isinstance(indices_or_sections, int) else indices_or_sections
    node: object = _emit_shape_node("Dsplit", [ary], {"indices_or_sections": indices_or_sections}, ary.shape, ary.dtype)
    out_tensors: object = []

    out_shapes: object = []
    if isinstance(indices_or_sections, int):
        s: object = list(ary.shape)
        if len(s) > 2:
            s[2] = s[2] // indices_or_sections
        out_shapes: object = [tuple(s)] * num_splits
    else:
        out_shapes: object = [ary.shape] * num_splits  #   # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    for i in range(num_splits):
        item_node: object = builder.TracingNodeBuilder.emit_tracing_node("GetItem", node, output_index=i, key=str(i))
        item_node._shape = out_shapes[i]
        item_node.config = TensorConfig(out_shapes[i], item_node.dtype, item_node.device)
        out_tensors.append(item_node)
    return tuple(out_tensors)


@register_op("GetItem")
class GetItemOp(OpDef):
    """Operation to retrieve an item from a tensor."""

    def infer_shape(self, x: object, output_index: int = 0, **kwargs: object) -> Sequence[int]:
        """Infer shape for Unstack.

        Args:
            x (object): The x parameter.
            output_index (int): The output_index parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple: Result.
        """
        # We don't have enough info here if x is a node, but we can just return None
        return getattr(x, "shape", ())


old_split: object = split


@register_op("Unstack")
class Unstack(OpDef):
    """Unstack op for shape inference."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape for Unstack.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        input_shape: object = args[0].shape
        axis: object = kwargs.get("axis", 0)
        return tuple([input_shape[:axis] + input_shape[axis + 1 :]] * input_shape[axis])
