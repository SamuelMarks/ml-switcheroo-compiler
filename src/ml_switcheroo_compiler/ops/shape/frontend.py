# pylint: disable=duplicate-code

"""Defines shape, memory, and movement operations for Tensor objects.

This module provides functions to manipulate tensor shapes, dimensions, and memory
layouts, supporting both eager execution (using NumPy) and lazy execution (by tracing
and emitting logical nodes to a graph)
"""

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import dispatch_op
from ml_switcheroo_compiler.ops.shape.dynamic_slicing import (
    dynamic_slice,
    dynamic_update_slice,
    update_slice,
)
from ml_switcheroo_compiler.ops.shape.indexing import (
    boolean_mask,
    gather,
    gather_nd,
    invert_permutation,
    searchsorted,
    select,
    take,
    take_along_axis,
    where,
)
from ml_switcheroo_compiler.ops.shape.joining import (
    append,
    column_stack,
    concatenate,
    dstack,
    hstack,
    stack,
    vstack,
)
from ml_switcheroo_compiler.ops.shape.manipulation import (
    atleast_1d,
    atleast_2d,
    atleast_3d,
    broadcast_arrays,
    broadcast_in_dim,
    broadcast_to,
    expand,
    expand_dims,
    flatten,
    moveaxis,
    permute,
    reshape,
    reverse,
    roll,
    squeeze,
    swapaxes,
    transpose,
    unflatten,
    unsqueeze,
    view,
)
from ml_switcheroo_compiler.ops.shape.misc import (
    argsort,
    image_resize,
    meshgrid,
    pad,
    repeat,
    sort,
    tile,
    top_k,
    tril,
    triu,
)
from ml_switcheroo_compiler.ops.shape.scatter import (
    scatter,
    scatter_add,
    scatter_nd,
    tensor_scatter_add,
    tensor_scatter_max,
    tensor_scatter_min,
    tensor_scatter_update,
)
from ml_switcheroo_compiler.ops.shape.slicing import slice, strided_slice
from ml_switcheroo_compiler.ops.shape.splitting import (
    array_split,
    dsplit,
    hsplit,
    split,
    unstack,
    vsplit,
)
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def argwhere(a: object) -> Tensor:
    """Find the indices of array elements that are non-zero, grouped by element.

    Args:
        a (object): Input data.

    Returns:
        Tensor: Indices of elements that are non-zero.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Argwhere", getattr(a, "data", a))
        return Tensor(data, TensorConfig(data.shape, "int64", getattr(a, "device", None)))
    return _emit_shape_node("Argwhere", [a], {}, (None, None), "int64")


def argpartition(a: object, kth: object, axis: int = -1, kind: str = "introselect", order: object = None) -> Tensor:
    """Perform an indirect partition along the given axis."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Argpartition", getattr(a, "data", a), kth, axis=axis, kind=kind, order=order)
        return Tensor(data, TensorConfig(data.shape, "int64", getattr(a, "device", None)))
    return _emit_shape_node(
        "Argpartition",
        [a, kth],
        {"axis": axis, "kind": kind, "order": order},
        getattr(a, "shape", ()),
        "int64",
    )


def partition(a: object, kth: object, axis: int = -1, kind: str = "introselect", order: object = None) -> Tensor:
    """Return a partitioned copy of an array."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Partition", getattr(a, "data", a), kth, axis=axis, kind=kind, order=order)
        return Tensor(
            data,
            TensorConfig(data.shape, getattr(a, "dtype", "float32"), getattr(a, "device", None)),
        )
    return _emit_shape_node(
        "Partition",
        [a, kth],
        {"axis": axis, "kind": kind, "order": order},
        getattr(a, "shape", ()),
        getattr(a, "dtype", "float32"),
    )


def compress(condition: object, a: object, axis: int = None, out: object = None) -> Tensor:
    """Return selected slices of an array along given axis."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Compress", condition, getattr(a, "data", a), axis=axis, out=out)
        return Tensor(
            data,
            TensorConfig(data.shape, getattr(a, "dtype", "float32"), getattr(a, "device", None)),
        )
    return _emit_shape_node(
        "Compress",
        [condition, a],
        {"axis": axis, "out": out},
        (None,),
        getattr(a, "dtype", "float32"),
    )


def delete(arr: object, obj: object, axis: int = None) -> Tensor:
    """Return a new array with sub-arrays along an axis deleted."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Delete", getattr(arr, "data", arr), getattr(obj, "data", obj), axis=axis)
        return Tensor(
            data,
            TensorConfig(data.shape, getattr(arr, "dtype", "float32"), getattr(arr, "device", None)),
        )
    return _emit_shape_node("Delete", [arr, obj], {"axis": axis}, (None,), getattr(arr, "dtype", "float32"))


def diff(a: object, n: int = 1, axis: int = -1, prepend: object = None, append: object = None) -> Tensor:
    """Calculate the n-th discrete difference along the given axis."""
    if config.eager_mode:
        kwargs = {"n": n, "axis": axis}
        if prepend is not None:
            kwargs["prepend"] = prepend
        if append is not None:
            kwargs["append"] = append
        data = get_active_backend().execute_op("Diff", getattr(a, "data", a), **kwargs)
        return Tensor(
            data,
            TensorConfig(data.shape, getattr(a, "dtype", "float32"), getattr(a, "device", None)),
        )
    return _emit_shape_node(
        "Diff",
        [a],
        {"n": n, "axis": axis, "prepend": prepend, "append": append},
        (None,),
        getattr(a, "dtype", "float32"),
    )


def digitize(x: object, bins: object, right: bool = False) -> Tensor:
    """Return the indices of the bins to which each value in input array belongs."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Digitize", getattr(x, "data", x), getattr(bins, "data", bins), right=right)
        return Tensor(data, TensorConfig(data.shape, "int64", getattr(x, "device", None)))
    return _emit_shape_node("Digitize", [x, bins], {"right": right}, getattr(x, "shape", ()), "int64")


def choose(a: object, choices: object, out: object = None, mode: str = "raise") -> Tensor:
    """Construct an array from an index array and a list of arrays to choose from."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Choose", getattr(a, "data", a), choices, out=out, mode=mode)
        return Tensor(
            data,
            TensorConfig(data.shape, getattr(a, "dtype", "float32"), getattr(a, "device", None)),
        )
    return _emit_shape_node(
        "Choose",
        [a, choices],
        {"out": out, "mode": mode},
        getattr(a, "shape", ()),
        getattr(a, "dtype", "float32"),
    )


def diagonal(a: object, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Tensor:
    """Return specified diagonals."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Diagonal", getattr(a, "data", a), offset=offset, axis1=axis1, axis2=axis2)
        return Tensor(
            data,
            TensorConfig(data.shape, getattr(a, "dtype", "float32"), getattr(a, "device", None)),
        )
    return _emit_shape_node(
        "Diagonal",
        [a],
        {"offset": offset, "axis1": axis1, "axis2": axis2},
        (None,),
        getattr(a, "dtype", "float32"),
    )


def diagflat(v: object, k: int = 0) -> Tensor:
    """Create a two-dimensional array with the flattened input as a diagonal."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Diagflat", getattr(v, "data", v), k=k)
        return Tensor(
            data,
            TensorConfig(data.shape, getattr(v, "dtype", "float32"), getattr(v, "device", None)),
        )
    return _emit_shape_node("Diagflat", [v], {"k": k}, (None, None), getattr(v, "dtype", "float32"))


def diag_indices(n: int, ndim: int = 2) -> tuple[Tensor, ...]:
    """Return the indices to access the main diagonal of an array."""
    if config.eager_mode:
        data = get_active_backend().execute_op("DiagIndices", n, ndim=ndim)
        return tuple(Tensor(d, TensorConfig(d.shape, "int64", None)) for d in data)
    return (_emit_shape_node("DiagIndices", [n], {"ndim": ndim}, (None,), "int64"),)


def diag_indices_from(arr: object) -> tuple[Tensor, ...]:
    """Return the indices to access the main diagonal of an n-dimensional array."""
    if config.eager_mode:
        data = get_active_backend().execute_op("DiagIndicesFrom", getattr(arr, "data", arr))
        return tuple(Tensor(d, TensorConfig(d.shape, "int64", getattr(arr, "device", None))) for d in data)
    return (_emit_shape_node("DiagIndicesFrom", [arr], {}, (None,), "int64"),)


def dynamic_partition(data: Tensor, partitions: Tensor, num_partitions: int) -> list[Tensor]:
    """dynamic_partition."""
    # Returns list of tensors, but IR node produces multiple outputs or a list.
    # For now, we'll return what dispatch_op returns.
    return dispatch_op("DynamicPartition", data, partitions, num_partitions=num_partitions)


def dynamic_stitch(indices: list[Tensor], data: list[Tensor]) -> Tensor:
    """dynamic_stitch."""
    return dispatch_op("DynamicStitch", indices, data)


def tensor_scatter_sub(tensor: Tensor, indices: Tensor, updates: Tensor) -> Tensor:
    """tensor_scatter_sub."""
    return dispatch_op("TensorScatterSub", tensor, indices, updates)


def extract_volume_patches(input: Tensor, ksizes: list[int], strides: list[int], padding: str) -> Tensor:
    """extract_volume_patches."""
    return dispatch_op("ExtractVolumePatches", input, ksizes=ksizes, strides=strides, padding=padding)


def unravel_index(indices: Tensor, dims: Tensor) -> Tensor:
    """unravel_index."""
    return dispatch_op("UnravelIndex", indices, dims)


def dynamic_shape(x: Tensor) -> Tensor:
    """Returns the dynamic shape of the tensor.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: The dynamic shape.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("DynamicShape", x.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, DType.Int32, x.device),
        )
    return _emit_shape_node("DynamicShape", [x], {}, (len(x.shape),), DType.Int32)


__all__ = [
    "_emit_shape_node",
    "append",
    "argpartition",
    "argsort",
    "argwhere",
    "array_split",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "boolean_mask",
    "broadcast_arrays",
    "broadcast_in_dim",
    "broadcast_to",
    "choose",
    "column_stack",
    "compress",
    "concatenate",
    "delete",
    "diag_indices",
    "diag_indices_from",
    "diagflat",
    "diagonal",
    "diff",
    "digitize",
    "dsplit",
    "dstack",
    "dynamic_partition",
    "dynamic_slice",
    "dynamic_stitch",
    "dynamic_update_slice",
    "expand",
    "expand_dims",
    "extract_volume_patches",
    "flatten",
    "gather",
    "gather_nd",
    "hsplit",
    "hstack",
    "image_resize",
    "invert_permutation",
    "meshgrid",
    "moveaxis",
    "pad",
    "partition",
    "permute",
    "repeat",
    "reshape",
    "reverse",
    "roll",
    "scatter",
    "scatter_add",
    "scatter_nd",
    "searchsorted",
    "select",
    "slice",
    "sort",
    "split",
    "squeeze",
    "stack",
    "strided_slice",
    "swapaxes",
    "take",
    "take_along_axis",
    "tensor_scatter_add",
    "tensor_scatter_max",
    "tensor_scatter_min",
    "tensor_scatter_sub",
    "tensor_scatter_update",
    "tile",
    "top_k",
    "transpose",
    "tril",
    "triu",
    "unflatten",
    "unravel_index",
    "unsqueeze",
    "unstack",
    "update_slice",
    "view",
    "vsplit",
    "vstack",
    "where",
]
