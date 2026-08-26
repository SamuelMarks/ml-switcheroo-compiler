"""Module indexing.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Shape operations for Tensor objects."""
from dataclasses import dataclass

# pylint: disable=duplicate-code
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


@dataclass
class IndexSpec:
    """Specifies the parameters for an indexing operation.

    Attributes:
        start_index: The starting index for the slice.
        limit_index: The ending index for the slice (exclusive).
        stride: The step size between elements.
        axis: The axis along which to apply the slice.
    """

    start_index: int
    limit_index: int
    stride: int = 1
    axis: int = 0


def gather(input: Tensor, axis: int, index: Tensor):
    """Gather values along an axis specified by axis using index tensor.

    Args:
        input (Tensor): The source tensor
        axis (int): The axis along which to index
        index (Tensor): The indices of elements to gather

    Returns:
        Tensor: The gathered tensor.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "TakeAlongAxis",
            (input.data if type(input).__name__ == "Tensor" else input),
            (index.data if type(index).__name__ == "Tensor" else index),
            axis=axis,
        )
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))
    inputs = [input, index]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Gather",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def gather_nd(input: Tensor, indices: Tensor):
    """Gather slices from input tensor using multi-dimensional indices.

    Args:
        input (Tensor): The source tensor
        indices (Tensor): Index tensor of shape where the last dimension contains
            indices into input.

    Returns:
        Tensor: The gathered tensor.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "GatherNd",
            (input.data if type(input).__name__ == "Tensor" else input),
            (indices.data if type(indices).__name__ == "Tensor" else indices),
        )
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))
    inputs = [input, indices]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "GatherNd",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def take(input: Tensor, indices: Tensor, axis: int | None = None):
    """Take elements from the input tensor at the specified flat indices.

    Args:
        input (Tensor): The input tensor
        indices (Tensor): The flat indices of elements to take
        axis (Optional[int]): The axis to take along. Defaults to None.

    Returns:
        Tensor: A 1D tensor containing the selected elements.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "Take",
            (input.data if type(input).__name__ == "Tensor" else input),
            (indices.data if type(indices).__name__ == "Tensor" else indices),
            axis=axis,
        )
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))
    inputs = [input, indices]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Take",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def take_along_axis(arr, indices, axis: int):
    """Take values from the input array along a specified axis using 1D indices.

    Args:
        arr (object): The source array or tensor
        indices (object): The indices to take along the axis
        axis (int): The axis along which to take values

    Returns: Tensor: The selected values.
    """
    backend = get_active_backend()
    return backend.execute_op(
        "TakeAlongAxis",
        ((arr.data if type(arr).__name__ == "Tensor" else arr) if hasattr(arr, "device") else arr),
        ((indices.data if type(indices).__name__ == "Tensor" else indices) if hasattr(indices, "device") else indices),
        axis=axis,
    )


def searchsorted(a: Tensor, v: Tensor, side: str = "left"):
    """Find indices where elements should be inserted to maintain order.

    Args:
        a (Tensor): 1-D input array. If side is 'left' or 'right',
            it must be sorted in ascending order.
        v (Tensor): Values to insert into a.
        side (str): If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.

    Returns:
        Tensor: Array of insertion points with the same shape as v.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "Searchsorted",
            (a.data if type(a).__name__ == "Tensor" else a),
            (v.data if type(v).__name__ == "Tensor" else v),
            side=side,
        )

        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, a.device))

    inputs = [a, v]
    attributes = {"side": side}

    return _emit_shape_node("SearchSorted", inputs, attributes, v.shape, DType.Int32)


def where(condition: Tensor, input: Tensor, other: Tensor):
    """Select elements from input or other based on condition.

    Args:
        condition (Tensor): A boolean tensor where True selects from input and False
            from other.
        input (Tensor): The tensor to select from where condition is True.
        other (Tensor): The tensor to select from where condition is False.

    Returns:
        Tensor: The selected tensor.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "Where",
            (condition.data if type(condition).__name__ == "Tensor" else condition),
            (input.data if type(input).__name__ == "Tensor" else input),
            (other.data if type(other).__name__ == "Tensor" else other),
        )
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))
    inputs = [condition, input, other]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Where",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def select(pred: Tensor, on_true: Tensor, on_false: Tensor):
    """Select elements from on_true or on_false based on pred.

    Args:
        pred (Tensor): Boolean mask
        on_true (Tensor): Selected when pred is True
        on_false (Tensor): Selected when pred is False

    Returns:
        Tensor: Resulting tensor.
    """
    return where(pred, on_true, on_false)


def boolean_mask(tensor: Tensor, mask: Tensor, axis: int | None = None):
    """Apply boolean mask to tensor.

    Args:
        tensor (Tensor): N-D tensor.
        mask (Tensor): K-D boolean tensor, K <= N.
        axis (int | None): A 0-D int Tensor representing the axis in tensor to mask from.

    Returns:
        Tensor: (N-K+1)-dimensional tensor populated by entries in tensor
        corresponding to True values in mask.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "BooleanMask",
            (tensor.data if type(tensor).__name__ == "Tensor" else tensor),
            (mask.data if type(mask).__name__ == "Tensor" else mask),
            axis=axis,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, tensor.dtype, tensor.device),
        )
    inputs = [tensor, mask]
    attributes = {"axis": axis}
    out_shape = (None,) * (len(tensor.shape) - len(mask.shape) + 1)
    return _emit_shape_node(
        "BooleanMask",
        inputs,
        attributes,
        out_shape,
        tensor.dtype,
    )


def invert_permutation(x: Tensor):
    """Compute the inverse permutation of a tensor.

    Args:
        x (Tensor): 1-D int32 or int64 tensor.

    Returns:
        Tensor: 1-D tensor of the same type as x.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("InvertPermutation", (x.data if type(x).__name__ == "Tensor" else x))
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, x.dtype, x.device),
        )
    inputs = [x]
    out_shape = x.shape
    return _emit_shape_node(
        "InvertPermutation",
        inputs,
        {},
        out_shape,
        x.dtype,
    )


@register_op("Extract")
class Extract(OpDef):
    """Extract operator."""

    op_name = "Extract"

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for the infer_shape operation.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        # Typically returns a 1D tensor of elements satisfying a condition.
        return (None,)


@register_op("DynamicPartition")
class DynamicPartition(OpDef):
    """Dynamic partition operation."""

    op_name = "DynamicPartition"

    def infer_shape(self, data, partitions, num_partitions: int, **kwargs):
        """Infers the output shape for the dynamic partition operation.

        Args:
            data (object): The tensor to be partitioned.
            partitions (object): A tensor containing partition indices.
            num_partitions (int): The total number of output partitions.
            **kwargs (object): Additional keyword arguments.

        Returns: Tensor: An empty tuple representing a placeholder shape for multiple outputs.
        """
        # returns list of tensors, hard to represent simply here
        return ()


@register_op("DynamicStitch")
class DynamicStitch(OpDef):
    """Dynamic stitch operation."""

    op_name = "DynamicStitch"

    def infer_shape(self, indices, data, **kwargs):
        """Infers the output shape for the dynamic stitch operation.

        Args:
            indices (object): The indices tensor.
            data (object): The data tensor.
            **kwargs (object): Additional keyword arguments.

        Returns: Tensor: An empty tuple representing a placeholder shape.
        """
        return ()


@register_op("TensorScatterSub")
class TensorScatterSub(OpDef):
    """Tensor scatter subtraction operation."""

    op_name = "TensorScatterSub"

    def infer_shape(self, tensor, indices, updates, **kwargs):
        """Infers the output shape for the tensor scatter sub operation.

        Args:
            tensor (object): The input tensor.
            indices (object): The indices tensor.
            updates (object): The updates tensor.
            **kwargs (object): Additional keyword arguments.

        Returns: Tensor: The shape of the input tensor.
        """
        return getattr(tensor, "shape", ())


@register_op("ExtractVolumePatches")
class ExtractVolumePatches(OpDef):
    """Extract volume patches operation."""

    op_name = "ExtractVolumePatches"

    def infer_shape(self, input, ksizes: list[int], strides: list[int], padding: str, **kwargs):
        """Infers the output shape for the extract volume patches operation.

        Args:
            input (object): The input tensor.
            ksizes (list[int]): The size of the sliding window.
            strides (list[int]): How far the centers of two consecutive patches are in the input.
            padding (str): The type of padding algorithm to use.
            **kwargs (object): Additional keyword arguments.

        Returns: Tensor: An empty tuple representing a placeholder shape.
        """
        return ()


@register_op("UnravelIndex")
class UnravelIndex(OpDef):
    """Unravel index operation."""

    op_name = "UnravelIndex"

    def infer_shape(self, indices, dims, **kwargs):
        """Infers the output shape for the unravel index operation.

        Args:
            indices (object): The indices tensor.
            dims (object): The dimensions tensor.
            **kwargs (object): Additional keyword arguments.

        Returns: Tensor: An empty tuple representing a placeholder shape for multiple outputs.
        """
        # unravel_index returns a tuple of tensors
        return ()


@register_op("DynamicSliceInDim")
class DynamicSliceInDim(OpDef):
    """Dynamic slice in dimension operator."""

    op_name = "DynamicSliceInDim"

    def infer_shape(self, operand, start_index, slice_size: int, axis: int = 0, **kwargs):
        """Infers the output shape for the dynamic slice in axis operation.

        Args:
            operand (object): The operand parameter.
            start_index (object): The start_index parameter.
            slice_size (int): The slice_size parameter.
            axis (int): The axis parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        shape = list(getattr(operand, "shape", ()))
        if shape:
            shape[axis] = slice_size
        return tuple(shape)


@register_op("DynamicUpdateSliceInDim")
class DynamicUpdateSliceInDim(OpDef):
    """Dynamic update slice in dimension operator."""

    op_name = "DynamicUpdateSliceInDim"

    def infer_shape(self, operand, update, start_index, axis: int = 0, **kwargs):
        """Infers the output shape for the dynamic update slice in axis operation.

        Args:
            operand (object): The operand parameter.
            update (object): The update parameter.
            start_index (object): The start_index parameter.
            axis (int): The axis parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(operand, "shape", ())


@register_op("DynamicIndexInDim")
class DynamicIndexInDim(OpDef):
    """Dynamic index in dimension operator."""

    op_name = "DynamicIndexInDim"

    def infer_shape(self, operand, index, axis: int = 0, keepdims: bool = True, **kwargs):
        """Infers the output shape for the dynamic index in axis operation.

        Args:
            operand (object): The operand parameter.
            index (object): The index parameter.
            axis (int): The axis parameter.
            keepdims (bool): The keepdims parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        shape = list(getattr(operand, "shape", ()))
        if shape:
            if keepdims:
                shape[axis] = 1
            else:
                shape.pop(axis)
        return tuple(shape)


@register_op("DynamicUpdateIndexInDim")
class DynamicUpdateIndexInDim(OpDef):
    """Dynamic update index in dimension operator."""

    op_name = "DynamicUpdateIndexInDim"

    def infer_shape(self, operand, update, index, axis: int = 0, **kwargs):
        """Infers the output shape for the dynamic update index in axis operation.

        Args:
            operand (object): The operand parameter.
            update (object): The update parameter.
            index (object): The index parameter.
            axis (int): The axis parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(operand, "shape", ())


@register_op("SliceInDim")
class SliceInDim(OpDef):
    """Slice in dimension operator."""

    op_name = "SliceInDim"

    def infer_shape(
        self,
        operand,
        spec: IndexSpec,
        **kwargs,
    ):
        """Infers the output shape for the slice in axis operation.

        Args:
            operand (object): The input tensor.
            spec (IndexSpec): The index specification.
            **kwargs (object): Additional keyword arguments.

        Returns: Tensor: The updated shape tuple.
        """
        shape = list(getattr(operand, "shape", ()))
        if shape:
            shape[spec.axis] = (spec.limit_index - spec.start_index + spec.stride - 1) // spec.stride
        return tuple(shape)


@register_op("ScatterApply")
class ScatterApply(OpDef):
    """Scatter apply operator."""

    op_name = "ScatterApply"

    def infer_shape(self, tensor, indices, updates, func, **kwargs):
        """Infers the output shape for the scatter apply operation.

        Args:
            tensor (object): The input tensor.
            indices (object): The indices tensor.
            updates (object): The updates tensor.
            func (object): The function to apply.
            **kwargs (object): Additional keyword arguments.

        Returns: Tensor: The shape of the input tensor.
        """
        return getattr(tensor, "shape", ())


@register_op("ScatterMax")
class ScatterMax(OpDef):
    """Scatter max operator."""

    op_name = "ScatterMax"

    def infer_shape(self, tensor, indices, updates, **kwargs):
        """Infers the output shape for the scatter max operation.

        Args:
            tensor (object): The input tensor.
            indices (object): The indices tensor.
            updates (object): The updates tensor.
            **kwargs (object): Additional keyword arguments.

        Returns: Tensor: The shape of the input tensor.
        """
        return getattr(tensor, "shape", ())


@register_op("ScatterMin")
class ScatterMin(OpDef):
    """Scatter min operator."""

    op_name = "ScatterMin"

    def infer_shape(self, tensor, indices, updates, **kwargs):
        """Infers the output shape for the scatter min operation.

        Args:
            tensor (object): The input tensor.
            indices (object): The indices tensor.
            updates (object): The updates tensor.
            **kwargs (object): Additional keyword arguments.

        Returns: Tensor: The shape of the input tensor.
        """
        return getattr(tensor, "shape", ())


@register_op("ScatterMul")
class ScatterMul(OpDef):
    """Scatter multiply operator."""

    op_name = "ScatterMul"

    def infer_shape(self, tensor, indices, updates, **kwargs):
        """Infers the output shape for the scatter mul operation.

        Args:
            tensor (object): The input tensor.
            indices (object): The indices tensor.
            updates (object): The updates tensor.
            **kwargs (object): Additional keyword arguments.

        Returns: Tensor: The shape of the input tensor.
        """
        return getattr(tensor, "shape", ())


@register_op("PutAlongAxis")
class PutAlongAxis(OpDef):
    """Put along axis operation."""

    op_name = "PutAlongAxis"

    def infer_shape(self, arr, indices, values, **kwargs):
        """Infers the output shape for the put along axis operation.

        Args:
            arr (object): The input array or tensor.
            indices (object): The indices tensor.
            values (object): The values tensor.
            **kwargs (object): Additional keyword arguments.

        Returns: Tensor: The shape of the input array or tensor.
        """
        return getattr(arr, "shape", ())


def put_along_axis(arr: Tensor, indices: Tensor, values: Tensor, axis: int):
    """Put values into array along axis at given indices.

    Args:
        arr (Tensor): The input array or tensor.
        indices (Tensor): The indices tensor.
        values (Tensor): The values to put.
        axis (int): The axis along which to put values.

    Returns:
        Tensor: The modified tensor.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "PutAlongAxis",
            (arr.data if type(arr).__name__ == "Tensor" else arr),
            (indices.data if type(indices).__name__ == "Tensor" else indices),
            (values.data if type(values).__name__ == "Tensor" else values),
            axis=axis,
        )
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", getattr(arr, "shape", ())),
                getattr(arr, "dtype", DType.Float32),
                getattr(arr, "device", config.default_device),
            ),
        )
    return _emit_shape_node(
        "PutAlongAxis",
        [arr, indices, values],
        {"axis": axis},
        getattr(arr, "shape", ()),
        getattr(arr, "dtype", DType.Float32),
    )
