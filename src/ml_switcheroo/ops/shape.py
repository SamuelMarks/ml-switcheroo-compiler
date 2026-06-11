"""Shape, Memory, and Movement Ops."""

import uuid
from typing import Union, Optional
from collections.abc import Sequence
import numpy as np
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.config import config
from ml_switcheroo.core.errors import UnimplementedMathError
from ml_switcheroo.tracing import _tracer, ProxyTensor
from ml_switcheroo_ir import LogicalNode


def _emit_shape_node(
    op_type: str,
    inputs: Sequence[Tensor],
    attrs: dict,
    out_shape: tuple,
    out_dtype: DType,
) -> Tensor:
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=[inp.data.id for inp in inputs],
        attributes=attrs,
        shape_metadata=out_shape,
    )
    _tracer.add_node(node)
    proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype.value)
    device = inputs[0].device if len(inputs) > 0 else config.default_device
    return Tensor(data=proxy, shape=out_shape, dtype=out_dtype, device=device)


def reshape(input: Tensor, shape: Sequence[int]) -> Tensor:
    """Reshape."""
    if config.eager_mode:
        data = np.reshape(input.data, shape)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        inputs = [input]
        # shape calculation placeholder
        out_shape = tuple(shape)
        return _emit_shape_node(
            "Reshape",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def flatten(input: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    """Flatten."""
    if config.eager_mode:
        data = np.reshape(input.data, -1)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        inputs = [input]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "Flatten",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def squeeze(input: Tensor, dim: Optional[Union[int, Sequence[int]]] = None) -> Tensor:
    """Squeeze."""
    if config.eager_mode:
        data = np.squeeze(input.data, axis=dim)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        inputs = [input]
        if dim is None:
            out_shape = tuple(s for s in input.shape if s != 1)
        else:
            dims = [dim] if isinstance(dim, int) else dim
            out_shape = tuple(
                s for i, s in enumerate(input.shape) if i not in dims or s != 1
            )
        return _emit_shape_node(
            "Squeeze",
            inputs,
            {"dim": dim} if dim is not None else {},
            out_shape,
            input.dtype,
        )


def unsqueeze(input: Tensor, dim: int) -> Tensor:
    """Unsqueeze."""
    if config.eager_mode:
        data = np.expand_dims(input.data, axis=dim)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        inputs = [input]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "Unsqueeze",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def expand(input: Tensor, size: Sequence[int]) -> Tensor:
    """Expand."""
    if config.eager_mode:
        data = np.broadcast_to(input.data, size)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        inputs = [input]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "Expand",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def broadcast_to(input: Tensor, size: Sequence[int]) -> Tensor:
    """broadcast_to."""
    if config.eager_mode:
        data = np.broadcast_to(input.data, size)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        inputs = [input]
        # shape calculation placeholder
        out_shape = tuple(size)
        return _emit_shape_node(
            "BroadcastTo",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def transpose(input: Tensor, dim0: int, dim1: int) -> Tensor:
    """Transpose."""
    if config.eager_mode:
        data = np.swapaxes(input.data, dim0, dim1)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        inputs = [input]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "Transpose",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def permute(input: Tensor, dims: Sequence[int]) -> Tensor:
    """Permute."""
    if config.eager_mode:
        data = np.transpose(input.data, dims)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        inputs = [input]
        # shape calculation placeholder
        out_shape = tuple(input.shape[d] for d in dims)
        return _emit_shape_node(
            "Permute",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def swapaxes(input: Tensor, axis1: int, axis2: int) -> Tensor:
    """Swapaxes."""
    if config.eager_mode:
        data = np.swapaxes(input.data, axis1, axis2)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        inputs = [input]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "Swapaxes",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def moveaxis(
    input: Tensor,
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
) -> Tensor:
    """Moveaxis."""
    if config.eager_mode:
        data = np.moveaxis(input.data, source, destination)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        inputs = [input]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "Moveaxis",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def roll(
    input: Tensor,
    shifts: Union[int, Sequence[int]],
    dims: Optional[Union[int, Sequence[int]]] = None,
) -> Tensor:
    """Roll."""
    if config.eager_mode:
        data = np.roll(input.data, shifts, axis=dims)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        inputs = [input]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "Roll",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def slice(
    input: Tensor,
    dim: int,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
) -> Tensor:
    """Slice."""
    if config.eager_mode:
        import builtins

        sl = [builtins.slice(None)] * len(input.shape)
        sl[dim] = builtins.slice(start, end, step)
        data = input.data[tuple(sl)]
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        inputs = [input]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "Slice",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def dynamic_slice(
    input: Tensor, start_indices: Sequence[Tensor], slice_sizes: Sequence[int]
) -> Tensor:
    """dynamic_slice."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct numpy for dynamic_slice")
    else:
        inputs = [input]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "DynamicSlice",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def update_slice(input: Tensor, update: Tensor, start_indices: Sequence[int]) -> Tensor:
    """update_slice."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct numpy for update_slice")
    else:
        inputs = [input, update]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "UpdateSlice",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def strided_slice(
    input: Tensor, begin: Sequence[int], end: Sequence[int], strides: Sequence[int]
) -> Tensor:
    """strided_slice."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct numpy for strided_slice")
    else:
        inputs = [input]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "StridedSlice",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def concatenate(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    """Concatenate."""
    if config.eager_mode:
        data = np.concatenate([t.data for t in tensors], axis=dim)
        return Tensor(data, data.shape, tensors[0].dtype, tensors[0].device)
    else:
        inputs = list(tensors)
        # shape calculation placeholder
        out_shape = tuple(
            sum(t.shape[i] for t in tensors) if i == dim else tensors[0].shape[i]
            for i in range(len(tensors[0].shape))
        )
        return _emit_shape_node(
            "Concatenate",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    """Stack."""
    if config.eager_mode:
        data = np.stack([t.data for t in tensors], axis=dim)
        return Tensor(data, data.shape, tensors[0].dtype, tensors[0].device)
    else:
        inputs = list(tensors)
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "Stack",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def split(
    input: Tensor, split_size_or_sections: Union[int, Sequence[int]], dim: int = 0
) -> Sequence[Tensor]:
    """Split."""
    if config.eager_mode:
        datas = np.split(input.data, split_size_or_sections, axis=dim)
        return tuple(Tensor(d, d.shape, input.dtype, input.device) for d in datas)
    else:
        inputs = [input]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return (
            _emit_shape_node(
                "Split",
                inputs,
                {},
                out_shape,
                inputs[0].dtype if len(inputs) > 0 else DType.Float32,
            ),
        )


def unstack(input: Tensor, dim: int = 0) -> Sequence[Tensor]:
    """Unstack."""
    if config.eager_mode:
        datas = (
            np.unstack(input.data, axis=dim)
            if hasattr(np, "unstack")
            else np.moveaxis(input.data, dim, 0)
        )
        return tuple(Tensor(d, d.shape, input.dtype, input.device) for d in datas)
    else:
        inputs = [input]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return (
            _emit_shape_node(
                "Unstack",
                inputs,
                {},
                out_shape,
                inputs[0].dtype if len(inputs) > 0 else DType.Float32,
            ),
        )


def tile(input: Tensor, reps: Sequence[int]) -> Tensor:
    """Tile."""
    if config.eager_mode:
        data = np.tile(input.data, reps)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        inputs = [input]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "Tile",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def repeat(
    input: Tensor, repeats: Union[int, Sequence[int]], dim: Optional[int] = None
) -> Tensor:
    """Repeat."""
    if config.eager_mode:
        data = np.repeat(input.data, repeats, axis=dim)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        inputs = [input]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "Repeat",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def gather(input: Tensor, dim: int, index: Tensor) -> Tensor:
    """Gather."""
    if config.eager_mode:
        data = np.take_along_axis(input.data, index.data, axis=dim)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        inputs = [input, index]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "Gather",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def gather_nd(input: Tensor, indices: Tensor) -> Tensor:
    """gather_nd."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct numpy for gather_nd")
    else:
        inputs = [input, indices]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "GatherNd",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def scatter(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    """Scatter."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct numpy for scatter")
    else:
        inputs = [input, index, src]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "Scatter",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def scatter_nd(indices: Tensor, updates: Tensor, shape: Sequence[int]) -> Tensor:
    """scatter_nd."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct numpy for scatter_nd")
    else:
        inputs = [indices, updates]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "ScatterNd",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def scatter_add(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    """scatter_add."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct numpy for scatter_add")
    else:
        inputs = [input, index, src]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "ScatterAdd",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def take(input: Tensor, indices: Tensor) -> Tensor:
    """Take."""
    if config.eager_mode:
        data = np.take(input.data, indices.data)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        inputs = [input, indices]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "Take",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def where(condition: Tensor, input: Tensor, other: Tensor) -> Tensor:
    """Where."""
    if config.eager_mode:
        data = np.where(condition.data, input.data, other.data)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        inputs = [condition, input, other]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "Where",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def triu(input: Tensor, diagonal: int = 0) -> Tensor:
    """Triu."""
    if config.eager_mode:
        data = np.triu(input.data, k=diagonal)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        inputs = [input]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "Triu",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def tril(input: Tensor, diagonal: int = 0) -> Tensor:
    """Tril."""
    if config.eager_mode:
        data = np.tril(input.data, k=diagonal)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        inputs = [input]
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return _emit_shape_node(
            "Tril",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )


def meshgrid(*tensors: Tensor, indexing: str = "ij") -> Sequence[Tensor]:
    """Meshgrid."""
    if config.eager_mode:
        datas = np.meshgrid(*[t.data for t in tensors], indexing=indexing)
        return tuple(
            Tensor(d, d.shape, tensors[0].dtype, tensors[0].device) for d in datas
        )
    else:
        inputs = list(tensors)
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        return tuple(
            _emit_shape_node(
                "Meshgrid",
                inputs,
                {},
                out_shape,
                inputs[0].dtype if len(inputs) > 0 else DType.Float32,
            )
            for _ in inputs
        )


from typing import Any


def pad(array: Any, pad_width: Any, mode: str = "constant", **kwargs) -> Any:
    import numpy as np

    return np.pad(array, pad_width, mode=mode, **kwargs)


def take_along_axis(arr: Any, indices: Any, axis: int) -> Any:
    import numpy as np

    return np.take_along_axis(
        (arr.data if hasattr(arr, "device") else arr),
        (indices.data if hasattr(indices, "device") else indices),
        axis=axis,
    )
