"""Shape operations for Tensor objects."""

from __future__ import annotations
# pylint: disable=duplicate-code


from typing import TYPE_CHECKING

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

if TYPE_CHECKING:
    from collections.abc import Sequence


def gather(input: Tensor, dim: int, index: Tensor) -> Tensor:  # pragma: no cover
    """Gathers values along an axis specified by dim using index tensor.

    Args:
        input (Tensor): The source tensor
        dim (int): The axis along which to index
        index (Tensor): The indices of elements to gather

    Returns:
    Tensor: The gathered tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("TakeAlongAxis", input.data, index.data, axis=dim)
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device)
        )
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


def gather_nd(input: Tensor, indices: Tensor) -> Tensor:  # pragma: no cover
    """Gathers slices from input tensor using multi-dimensional indices.

    Args:
        input (Tensor): The source tensor
        indices (Tensor): Index tensor of shape where the last dimension contains
        indices into input

    Returns:
    Tensor: The gathered tensor

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("GatherNd", input.data, indices.data)
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device)
        )
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


def take(input: Tensor, indices: Tensor, axis: int | None = None) -> Tensor:  # pragma: no cover
    """Takes elements from the input tensor at the specified flat indices.

    Args:
        input (Tensor): The input tensor
        indices (Tensor): The flat indices of elements to take
        axis (Optional[int]): The axis to take along. Defaults to None.

    Returns:
    Tensor: A 1D tensor containing the selected elements
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Take", input.data, indices.data, axis=axis)
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device)
        )
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


def take_along_axis(arr: object, indices: object, axis: int) -> object:  # pragma: no cover
    """Takes values from the input array along a specified axis using 1D indices.

    Args:
        arr (object): The source array or tensor
        indices (object): The indices to take along the axis
        axis (int): The axis along which to take values

    Returns:
    object: The selected values
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op(
        "TakeAlongAxis",
        (arr.data if hasattr(arr, "device") else arr),
        (indices.data if hasattr(indices, "device") else indices),
        axis=axis,
    )


def searchsorted(a: Tensor, v: Tensor, side: str = "left") -> Tensor:  # pragma: no cover
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
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Searchsorted", a.data, v.data, side=side)
        from ml_switcheroo_compiler.core.dtype import DType

        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, a.device)
        )

    inputs = [a, v]
    attributes = {"side": side}
    from ml_switcheroo_compiler.core.dtype import DType

    return _emit_shape_node("SearchSorted", inputs, attributes, v.shape, DType.Int32)


def scatter(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:  # pragma: no cover
    """Scatters values from a source tensor into the input tensor along a specified.

    dimension

    Args:
        input (Tensor): The destination tensor
        dim (int): The axis along which to index
        index (Tensor): The indices of elements to scatter
        src (Tensor): The source tensor containing values to scatter

    Returns:
    Tensor: The updated tensor

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Scatter", input.data, index.data, src.data, dim=dim)
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device)
        )
    inputs = [input, index, src]
    attributes = {"dim": dim}
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Scatter",
        inputs,
        attributes,
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def scatter_nd(
    indices: Tensor, updates: Tensor, shape: Sequence[int]
) -> Tensor:  # pragma: no cover
    """Scatters updates into a new tensor of specified shape using indices.

    Args:
        indices (Tensor): The index tensor
        updates (Tensor): The updates to scatter
        shape (Sequence[int]): The shape of the output tensor

    Returns:
    Tensor: The output tensor with scattered updates

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("ScatterNd", indices.data, updates.data, shape=shape)
        return Tensor(
            backend.array(data), TensorConfig(tuple(shape), updates.dtype, updates.device)
        )
    inputs = [indices, updates]
    attributes = {"shape": shape}
    return _emit_shape_node(
        "ScatterNd",
        inputs,
        attributes,
        tuple(shape),
        inputs[1].dtype if len(inputs) > 1 else DType.Float32,
    )


def scatter_add(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:  # pragma: no cover
    """Adds values from a source tensor to the input tensor at specified indices along a.

    dimension

    Args:
        input (Tensor): The destination tensor
        dim (int): The axis along which to index
        index (Tensor): The indices of elements to add
        src (Tensor): The source tensor containing values to add

    Returns:
    Tensor: The updated tensor

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("ScatterAdd", input.data, index.data, src.data, dim=dim)
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device)
        )
    inputs = [input, index, src]
    attributes = {"dim": dim}
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "ScatterAdd",
        inputs,
        attributes,
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def where(condition: Tensor, input: Tensor, other: Tensor) -> Tensor:  # pragma: no cover
    """Selects elements from input or other based on condition.

    Args:
        condition (Tensor): A boolean tensor where True selects from input and False
        from other
        input (Tensor): The tensor to select from where condition is True
        other (Tensor): The tensor to select from where condition is False

    Returns:
    Tensor: The selected tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Where", condition.data, input.data, other.data)
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device)
        )
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


def select(pred: Tensor, on_true: Tensor, on_false: Tensor) -> Tensor:  # pragma: no cover
    """Selects elements from on_true or on_false based on pred.

    Args:
        pred (Tensor): Boolean mask
        on_true (Tensor): Selected when pred is True
        on_false (Tensor): Selected when pred is False

    Returns:
    Tensor: Resulting tensor
    """
    return where(pred, on_true, on_false)


def tensor_scatter_update(
    tensor: Tensor, indices: Tensor, updates: Tensor
) -> Tensor:  # pragma: no cover
    """Updates the value of a tensor at given indices.

    Args:
        tensor (Tensor): The input tensor
        indices (Tensor): The indices to update
        updates (Tensor): The updates to apply

    Returns:
    Tensor: The updated tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("TensorScatterUpdate", tensor.data, indices.data, updates.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, tensor.dtype, tensor.device),
        )
    inputs = [tensor, indices, updates]
    # shape calculation placeholder
    out_shape = tensor.shape
    return _emit_shape_node(
        "TensorScatterUpdate",
        inputs,
        {},
        out_shape,
        tensor.dtype,
    )


def tensor_scatter_max(
    tensor: Tensor, indices: Tensor, updates: Tensor
) -> Tensor:  # pragma: no cover
    """Updates a tensor at given indices with the maximum of the current value and the update.

    Args:
        tensor (Tensor): The input tensor
        indices (Tensor): The indices to update
        updates (Tensor): The updates to apply

    Returns:
    Tensor: The updated tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("TensorScatterMax", tensor.data, indices.data, updates.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, tensor.dtype, tensor.device),
        )
    inputs = [tensor, indices, updates]
    # shape calculation placeholder
    out_shape = tensor.shape
    return _emit_shape_node(
        "TensorScatterMax",
        inputs,
        {},
        out_shape,
        tensor.dtype,
    )


def tensor_scatter_min(
    tensor: Tensor, indices: Tensor, updates: Tensor
) -> Tensor:  # pragma: no cover
    """Updates a tensor at given indices with the minimum of the current value and the update.

    Args:
        tensor (Tensor): The input tensor
        indices (Tensor): The indices to update
        updates (Tensor): The updates to apply

    Returns:
    Tensor: The updated tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("TensorScatterMin", tensor.data, indices.data, updates.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, tensor.dtype, tensor.device),
        )
    inputs = [tensor, indices, updates]
    # shape calculation placeholder
    out_shape = tensor.shape
    return _emit_shape_node(
        "TensorScatterMin",
        inputs,
        {},
        out_shape,
        tensor.dtype,
    )


def tensor_scatter_add(
    tensor: Tensor, indices: Tensor, updates: Tensor
) -> Tensor:  # pragma: no cover
    """Adds updates to a tensor at given indices.

    Args:
        tensor (Tensor): The input tensor
        indices (Tensor): The indices to update
        updates (Tensor): The updates to apply

    Returns:
    Tensor: The updated tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("TensorScatterAdd", tensor.data, indices.data, updates.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, tensor.dtype, tensor.device),
        )
    inputs = [tensor, indices, updates]
    # shape calculation placeholder
    out_shape = tensor.shape
    return _emit_shape_node(
        "TensorScatterAdd",
        inputs,
        {},
        out_shape,
        tensor.dtype,
    )


def boolean_mask(
    tensor: Tensor, mask: Tensor, axis: int | None = None
) -> Tensor:  # pragma: no cover
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
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("BooleanMask", tensor.data, mask.data, axis=axis)
        return Tensor(  # pragma: no cover
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


def invert_permutation(x: Tensor) -> Tensor:  # pragma: no cover
    """Computes the inverse permutation of a tensor.

    Args:
        x (Tensor): 1-D int32 or int64 tensor.

    Returns:
        Tensor: 1-D tensor of the same type as x.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("InvertPermutation", x.data)
        return Tensor(  # pragma: no cover
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


@register_op("DynamicPartition")
class DynamicPartition(OpDef):
    """DynamicPartition operation."""

    op_name = "DynamicPartition"

    def infer_shape(  # pragma: no cover
        self, data: object, partitions: object, num_partitions: int, **kwargs: object
    ) -> object:
        """Infer shape."""
        # returns list of tensors, hard to represent simply here
        return ()


@register_op("DynamicStitch")
class DynamicStitch(OpDef):
    """DynamicStitch operation."""

    op_name = "DynamicStitch"

    def infer_shape(
        self, indices: object, data: object, **kwargs: object
    ) -> object:  # pragma: no cover
        """Infer shape."""
        return ()


@register_op("TensorScatterSub")
class TensorScatterSub(OpDef):
    """TensorScatterSub operation."""

    op_name = "TensorScatterSub"

    def infer_shape(  # pragma: no cover
        self, tensor: object, indices: object, updates: object, **kwargs: object
    ) -> object:
        """Infer shape."""
        return getattr(tensor, "shape", ())


@register_op("ExtractVolumePatches")
class ExtractVolumePatches(OpDef):
    """ExtractVolumePatches operation."""

    op_name = "ExtractVolumePatches"

    def infer_shape(  # pragma: no cover
        self, input: object, ksizes: list[int], strides: list[int], padding: str, **kwargs: object
    ) -> object:
        """Infer shape."""
        return ()


@register_op("UnravelIndex")
class UnravelIndex(OpDef):
    """UnravelIndex operation."""

    op_name = "UnravelIndex"

    def infer_shape(
        self, indices: object, dims: object, **kwargs: object
    ) -> object:  # pragma: no cover
        """Infer shape."""
        # unravel_index returns a tuple of tensors
        return ()


@register_op("DynamicSliceInDim")
class DynamicSliceInDim(OpDef):
    """DynamicSliceInDim operator."""

    op_name = "DynamicSliceInDim"

    def infer_shape(  # pragma: no cover
        self, operand: object, start_index: object, slice_size: int, axis: int = 0, **kwargs: object
    ) -> object:
        """Infer shape."""
        shape = list(getattr(operand, "shape", ()))
        if shape:
            shape[axis] = slice_size
        return tuple(shape)


@register_op("DynamicUpdateSliceInDim")
class DynamicUpdateSliceInDim(OpDef):
    """DynamicUpdateSliceInDim operator."""

    op_name = "DynamicUpdateSliceInDim"

    def infer_shape(  # pragma: no cover
        self, operand: object, update: object, start_index: object, axis: int = 0, **kwargs: object
    ) -> object:
        """Infer shape."""
        return getattr(operand, "shape", ())


@register_op("DynamicIndexInDim")
class DynamicIndexInDim(OpDef):
    """DynamicIndexInDim operator."""

    op_name = "DynamicIndexInDim"

    def infer_shape(  # pragma: no cover
        self, operand: object, index: object, axis: int = 0, keepdims: bool = True, **kwargs: object
    ) -> object:
        """Infer shape."""
        shape = list(getattr(operand, "shape", ()))
        if shape:
            if keepdims:
                shape[axis] = 1
            else:
                shape.pop(axis)
        return tuple(shape)


@register_op("DynamicUpdateIndexInDim")
class DynamicUpdateIndexInDim(OpDef):
    """DynamicUpdateIndexInDim operator."""

    op_name = "DynamicUpdateIndexInDim"

    def infer_shape(  # pragma: no cover
        self, operand: object, update: object, index: object, axis: int = 0, **kwargs: object
    ) -> object:
        """Infer shape."""
        return getattr(operand, "shape", ())


@register_op("SliceInDim")
class SliceInDim(OpDef):
    """SliceInDim operator."""

    op_name = "SliceInDim"

    def infer_shape(  # pragma: no cover
        self,
        operand: object,
        start_index: int,
        limit_index: int,
        stride: int = 1,
        axis: int = 0,
        **kwargs: object,
    ) -> object:
        """Infer shape."""
        shape = list(getattr(operand, "shape", ()))
        if shape:
            shape[axis] = (limit_index - start_index + stride - 1) // stride
        return tuple(shape)


@register_op("ScatterApply")
class ScatterApply(OpDef):
    """ScatterApply operator."""

    op_name = "ScatterApply"

    def infer_shape(  # pragma: no cover
        self, tensor: object, indices: object, updates: object, func: object, **kwargs: object
    ) -> object:
        """Infer shape."""
        return getattr(tensor, "shape", ())


@register_op("ScatterMax")
class ScatterMax(OpDef):
    """ScatterMax operator."""

    op_name = "ScatterMax"

    def infer_shape(  # pragma: no cover
        self, tensor: object, indices: object, updates: object, **kwargs: object
    ) -> object:
        """Infer shape."""
        return getattr(tensor, "shape", ())


@register_op("ScatterMin")
class ScatterMin(OpDef):
    """ScatterMin operator."""

    op_name = "ScatterMin"

    def infer_shape(  # pragma: no cover
        self, tensor: object, indices: object, updates: object, **kwargs: object
    ) -> object:
        """Infer shape."""
        return getattr(tensor, "shape", ())


@register_op("ScatterMul")
class ScatterMul(OpDef):
    """ScatterMul operator."""

    op_name = "ScatterMul"

    def infer_shape(  # pragma: no cover
        self, tensor: object, indices: object, updates: object, **kwargs: object
    ) -> object:
        """Infer shape."""
        return getattr(tensor, "shape", ())


@register_op("PutAlongAxis")
class PutAlongAxis(OpDef):
    """PutAlongAxis operation."""

    op_name = "PutAlongAxis"

    def infer_shape(self, arr: object, indices: object, values: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(arr, "shape", ())


def put_along_axis(arr: Tensor, indices: Tensor, values: Tensor, axis: int) -> Tensor:
    """Put values into arr along axis at indices."""
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "PutAlongAxis",
            getattr(arr, "data", arr),
            getattr(indices, "data", indices),
            getattr(values, "data", values),
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
