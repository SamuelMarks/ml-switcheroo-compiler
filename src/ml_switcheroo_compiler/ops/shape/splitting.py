"""Shape operations for Tensor Anys."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

# pylint: disable=duplicate-code
from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node
from ml_switcheroo_compiler.tracing import builder


def _calculate_num_splits(input: TensorOrScalar, split_size_or_sections: int | Sequence[int], axis: int) -> TensorOrScalar:  #
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

    if getattr(input, "shape") and getattr(input, "shape")[axis] % split_size_or_sections == 0:
        return cast(TensorOrScalar, getattr(input, "shape")[axis] // split_size_or_sections)

    return split_size_or_sections


def _validate_split_axis(input: TensorOrScalar, axis: int) -> TensorOrScalar:  #
    """Validate the split axis against the input tensor shape.

    Args:
        input (Tensor): The input parameter.
        axis (int): The axis parameter.

    Returns:
        int: Result.

    Raises:
        ValueError: An exception.
    """
    shape = getattr(input, "shape", None)
    if shape is None:
        return axis
    rank = len(shape)
    if not (0 <= axis < rank or -rank <= axis < 0):
        raise ValueError(f"Split axis {axis} is out of bounds for tensor of rank {rank}.")
    return axis


# pylint: disable=too-many-locals


def _split_even(input: TensorOrScalar, split_size: int, axis: int, num_splits: int) -> TensorOrScalar:
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
        backend = get_active_backend()
        input_data = getattr(input, "data", input)
        datas = backend.execute_op("Split", input_data, split_size, axis=axis)
        input_dtype = getattr(input, "dtype", getattr(backend.array(input_data), "dtype"))
        input_device = getattr(input, "device", config.default_device)
        return tuple(Tensor(d, TensorConfig(getattr(d, "shape"), input_dtype, input_device)) for d in getattr(datas, "__iter__")())

    inputs = [input]
    out_shape = getattr(inputs[0], "shape") if len(inputs) > 0 else ()
    node = _emit_shape_node(
        "Split",
        inputs,
        {"split_size_or_sections": split_size, "axis": axis},
        out_shape,
        getattr(inputs[0], "dtype") if len(inputs) > 0 else DType.Float32,
    )

    out_tensors = []
    for i in range(num_splits):
        item_node = builder.TracingNodeBuilder.emit_tracing_node("GetItem", node, output_index=i, key=str(i))
        out_tensors.append(item_node)
    return tuple(out_tensors)


def _split_sections(  # pylint: disable=too-many-locals
    input: TensorOrScalar,
    sections: Sequence[int],
    axis: int,
    num_splits: int,  #
) -> TensorOrScalar:
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
        backend = get_active_backend()
        input_data = getattr(input, "data", input)
        datas = backend.execute_op("Split", input_data, sections, axis=axis)
        input_dtype = getattr(input, "dtype", getattr(backend.array(input_data), "dtype"))
        input_device = getattr(input, "device", config.default_device)
        return tuple(Tensor(d, TensorConfig(getattr(d, "shape"), input_dtype, input_device)) for d in getattr(datas, "__iter__")())

    inputs = [input]
    out_shape = getattr(inputs[0], "shape") if len(inputs) > 0 else ()
    node = _emit_shape_node(
        "Split",
        inputs,
        {"split_size_or_sections": sections, "axis": axis},
        out_shape,
        getattr(inputs[0], "dtype") if len(inputs) > 0 else DType.Float32,
    )

    out_tensors = []
    for i in range(num_splits):
        item_node = builder.TracingNodeBuilder.emit_tracing_node("GetItem", node, output_index=i, key=str(i))
        out_tensors.append(item_node)
    return tuple(out_tensors)


def split(
    input: TensorOrScalar,  #
    split_size_or_sections: int | Sequence[int],
    axis: int = 0,
) -> TensorOrScalar:
    """Split the input tensor into multiple sub-tensors.

    Args:
        input (Tensor): The input parameter.
        split_size_or_sections (int | Sequence[int]): The split_size_or_sections parameter.
        axis (int): The axis parameter.

    Returns:
        Sequence: Result.
    """
    valid_axis = _validate_split_axis(input, axis)
    num_splits = _calculate_num_splits(input, split_size_or_sections, cast("int", valid_axis))

    if isinstance(split_size_or_sections, int):
        return _split_even(input, split_size_or_sections, cast("int", valid_axis), cast("int", num_splits))
    return _split_sections(input, split_size_or_sections, cast("int", valid_axis), cast("int", num_splits))


def unstack(input: TensorOrScalar, axis: int = 0) -> TensorOrScalar:
    """Unstack the input tensor along a specified dimension into a sequence of tensors.

    Args:
        input (Tensor): The input parameter.
        axis (int): The axis parameter.

    Returns:
        Sequence: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        datas = backend.execute_op("Unstack", getattr(input, "data"), axis=axis) if hasattr(backend, "unstack") else backend.execute_op("Moveaxis", getattr(input, "data"), axis, 0)
        input_data = getattr(input, "data", input)
        input_dtype = getattr(input, "dtype", getattr(backend.array(input_data), "dtype"))
        input_device = getattr(input, "device", config.default_device)
        return tuple(Tensor(d, TensorConfig(getattr(d, "shape"), input_dtype, input_device)) for d in getattr(datas, "__iter__")())
    inputs = [input]
    base_shape = getattr(inputs[0], "shape") if len(inputs) > 0 else ()
    rank = len(base_shape)
    norm_axis = axis if axis >= 0 else axis + rank
    if rank > 0 and 0 <= norm_axis < rank:
        n_slices = int(base_shape[norm_axis])
        out_shape = base_shape[:norm_axis] + base_shape[norm_axis + 1 :]
    else:
        n_slices = 1
        out_shape = ()
    return tuple(
        _emit_shape_node(
            "Unstack",
            inputs,
            {"axis": axis, "index": i},
            out_shape,
            getattr(inputs[0], "dtype") if len(inputs) > 0 else DType.Float32,
        )
        for i in range(n_slices)
    )


def array_split(
    ary: TensorOrScalar,  #
    indices_or_sections: int | Sequence[int],
    axis: int = 0,
) -> TensorOrScalar:
    """Split an array into multiple sub-arrays.

    Args:
        ary (Tensor): The ary parameter.
        indices_or_sections (int | Sequence[int]): The indices_or_sections parameter.
        axis (int): The axis parameter.

    Returns:
        Sequence: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        datas = backend.execute_op("ArraySplit", getattr(ary, "data"), indices_or_sections, axis=axis)
        return tuple(Tensor(d, TensorConfig(getattr(d, "shape"), getattr(ary, "dtype"), getattr(ary, "device"))) for d in getattr(datas, "__iter__")())
    return (
        _emit_shape_node(
            "ArraySplit",
            [ary],
            {"indices_or_sections": indices_or_sections, "axis": axis},
            getattr(ary, "shape"),
            getattr(ary, "dtype"),
        ),
    )


def vsplit(ary: TensorOrScalar, indices_or_sections: int | Sequence[int]) -> TensorOrScalar:
    """Split an array into multiple sub-arrays vertically (row-wise).

    Args:
        ary (Tensor): The ary parameter.
        indices_or_sections (int | Sequence[int]): The indices_or_sections parameter.

    Returns:
        Sequence: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        datas = backend.execute_op("Vsplit", getattr(ary, "data"), indices_or_sections)
        return tuple(Tensor(d, TensorConfig(getattr(d, "shape"), getattr(ary, "dtype"), getattr(ary, "device"))) for d in getattr(datas, "__iter__")())

    num_splits = len(indices_or_sections) + 1 if not isinstance(indices_or_sections, int) else indices_or_sections
    node = _emit_shape_node("Vsplit", [ary], {"indices_or_sections": indices_or_sections}, getattr(ary, "shape"), getattr(ary, "dtype"))
    out_tensors = []

    # Calculate output shapes
    out_shapes = []
    if isinstance(indices_or_sections, int):
        s = list(getattr(ary, "shape"))
        if len(s) > 0:
            s[0] = s[0] // indices_or_sections
        out_shapes = [tuple(s)] * num_splits
    else:
        # Just approximate
        out_shapes = [getattr(ary, "shape")] * num_splits  #   # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    for i in range(num_splits):
        item_node = builder.TracingNodeBuilder.emit_tracing_node("GetItem", node, output_index=i, key=str(i))
        item_node._shape = out_shapes[i]
        item_node.config = TensorConfig(out_shapes[i], item_node.dtype, item_node.device)
        out_tensors.append(item_node)
    return tuple(out_tensors)


def hsplit(ary: TensorOrScalar, indices_or_sections: int | Sequence[int]) -> TensorOrScalar:
    """Split an array into multiple sub-arrays horizontally (column-wise).

    Args:
        ary (Tensor): The ary parameter.
        indices_or_sections (int | Sequence[int]): The indices_or_sections parameter.

    Returns:
        Sequence: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        datas = backend.execute_op("Hsplit", getattr(ary, "data"), indices_or_sections)
        return tuple(Tensor(d, TensorConfig(getattr(d, "shape"), getattr(ary, "dtype"), getattr(ary, "device"))) for d in getattr(datas, "__iter__")())

    num_splits = len(indices_or_sections) + 1 if not isinstance(indices_or_sections, int) else indices_or_sections
    node = _emit_shape_node("Hsplit", [ary], {"indices_or_sections": indices_or_sections}, getattr(ary, "shape"), getattr(ary, "dtype"))
    out_tensors = []

    out_shapes = []
    if isinstance(indices_or_sections, int):
        s = list(getattr(ary, "shape"))
        if len(s) > 1:
            s[1] = s[1] // indices_or_sections
        out_shapes = [tuple(s)] * num_splits
    else:
        out_shapes = [getattr(ary, "shape")] * num_splits  #   # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    for i in range(num_splits):
        item_node = builder.TracingNodeBuilder.emit_tracing_node("GetItem", node, output_index=i, key=str(i))
        item_node._shape = out_shapes[i]
        item_node.config = TensorConfig(out_shapes[i], item_node.dtype, item_node.device)
        out_tensors.append(item_node)
    return tuple(out_tensors)


def dsplit(ary: TensorOrScalar, indices_or_sections: int | Sequence[int]) -> TensorOrScalar:
    """Split array into multiple sub-arrays along the 3rd axis (depth).

    Args:
        ary (Tensor): The ary parameter.
        indices_or_sections (int | Sequence[int]): The indices_or_sections parameter.

    Returns:
        Sequence: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        datas = backend.execute_op("Dsplit", getattr(ary, "data"), indices_or_sections)
        return tuple(Tensor(d, TensorConfig(getattr(d, "shape"), getattr(ary, "dtype"), getattr(ary, "device"))) for d in getattr(datas, "__iter__")())

    num_splits = len(indices_or_sections) + 1 if not isinstance(indices_or_sections, int) else indices_or_sections
    node = _emit_shape_node("Dsplit", [ary], {"indices_or_sections": indices_or_sections}, getattr(ary, "shape"), getattr(ary, "dtype"))
    out_tensors = []

    out_shapes = []
    if isinstance(indices_or_sections, int):
        s = list(getattr(ary, "shape"))
        if len(s) > 2:
            s[2] = s[2] // indices_or_sections
        out_shapes = [tuple(s)] * num_splits
    else:
        out_shapes = [getattr(ary, "shape")] * num_splits  #   # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    for i in range(num_splits):
        item_node = builder.TracingNodeBuilder.emit_tracing_node("GetItem", node, output_index=i, key=str(i))
        item_node._shape = out_shapes[i]
        item_node.config = TensorConfig(out_shapes[i], item_node.dtype, item_node.device)
        out_tensors.append(item_node)
    return tuple(out_tensors)


@register_op("GetItem")
class GetItemOp(OpDef):
    """Operation to retrieve an item from a tensor."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer shape for Unstack.

        Args:
            *args (EagerTensorProtocol): Positional arguments (e.g. input tensor).
            **kwargs (EagerTensorProtocol): Keyword arguments.

        Returns:
            tuple[int, ...]: Inferred shape.
        """
        x = args[0] if args else None
        return tuple(getattr(x, "shape", ()))


old_split = split


@register_op("Unstack")
class Unstack(OpDef):
    """Unstack op for shape inference."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape for Unstack.

        Args:
            *args (EagerTensorProtocol): Positional args.
            **kwargs (EagerTensorProtocol): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        input_shape = args[0].shape
        axis = kwargs.get("axis", 0)
        return tuple([cast(EagerTensorProtocol, input_shape)[: int(cast(int, axis))] + cast(EagerTensorProtocol, input_shape)[int(cast(EagerTensorProtocol, axis) + 1) :]] * int(cast(int, getattr(input_shape, "__getitem__")(axis))))
