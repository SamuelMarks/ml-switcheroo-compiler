# pylint: disable=duplicate-code

"""Defines shape, memory, and movement operations for Tensor objects.

This module provides functions to manipulate tensor shapes, dimensions, and memory
layouts, supporting both eager execution (using NumPy) and lazy execution (by tracing
and emitting logical nodes to a graph)
"""

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def expand_dims(a: object, axis: int) -> Tensor:
    """Expand dimensions."""
    if config.eager_mode:
        data = get_active_backend().execute_op("ExpandDims", getattr(a, "data", a), axis=axis)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("ExpandDims")().infer_shape(a, axis=axis)
    return _emit_shape_node("ExpandDims", [a], {"axis": axis}, out_shape, getattr(a, "dtype", "float32"))


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


def insert(arr: object, obj: object, values: object, axis: int = None) -> Tensor:
    """Insert values along the given axis before the given indices."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Insert", getattr(arr, "data", arr), obj, getattr(values, "data", values), axis=axis)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(arr, "dtype", "float32"), getattr(arr, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op

    out_shape = get_op("Insert")().infer_shape(arr, obj=obj, values=values, axis=axis)
    return _emit_shape_node("Insert", [arr, values] if hasattr(values, "shape") else [arr], {"obj": obj, "axis": axis}, out_shape, getattr(arr, "dtype", "float32"))


def fill_diagonal(a: object, val: object, wrap: bool = False) -> Tensor:
    """Fill the main diagonal of the given array of any dimensionality."""
    if config.eager_mode:
        data = get_active_backend().execute_op("FillDiagonal", getattr(a, "data", a), getattr(val, "data", val), wrap=wrap)
        return Tensor(data, TensorConfig(getattr(a, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op

    out_shape = get_op("FillDiagonal")().infer_shape(a, val=val, wrap=wrap)
    return _emit_shape_node("FillDiagonal", [a, val] if hasattr(val, "shape") else [a], {"val": val, "wrap": wrap}, out_shape, getattr(a, "dtype", "float32"))


def moveaxis(a: object, source: object, destination: object) -> Tensor:
    """Move axes of a tensor."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Moveaxis", getattr(a, "data", a), source=source, destination=destination)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Moveaxis")().infer_shape(a, source=source, destination=destination)
    return _emit_shape_node("Moveaxis", [a], {"source": source, "destination": destination}, out_shape, getattr(a, "dtype", "float32"))


def permute(a: object, dims: object = None) -> Tensor:
    """Permute dimensions of a tensor."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Permute", getattr(a, "data", a), dims=dims)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Permute")().infer_shape(a, dims=dims)
    return _emit_shape_node("Permute", [a], {"dims": dims}, out_shape, getattr(a, "dtype", "float32"))


def swapaxes(a: object, axis1: int, axis2: int) -> Tensor:
    """Interchange two axes of an array."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Swapaxes", getattr(a, "data", a), axis1=axis1, axis2=axis2)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Swapaxes")().infer_shape(a, axis1=axis1, axis2=axis2)
    return _emit_shape_node("Swapaxes", [a], {"axis1": axis1, "axis2": axis2}, out_shape, getattr(a, "dtype", "float32"))


def roll(a: object, shift: object, axis: object = None) -> Tensor:
    """Roll array elements along a given axis."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Roll", getattr(a, "data", a), shift=shift, axis=axis)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Roll")().infer_shape(a, shift=shift, axis=axis)
    return _emit_shape_node("Roll", [a], {"shift": shift, "axis": axis}, out_shape, getattr(a, "dtype", "float32"))


def atleast_1d(a: object) -> Tensor:
    """Atleast 1d."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Atleast1d", getattr(a, "data", a))
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Atleast1d")().infer_shape(a)
    return _emit_shape_node("Atleast1d", [a], {}, out_shape, getattr(a, "dtype", "float32"))


def atleast_2d(a: object) -> Tensor:
    """Atleast 2d."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Atleast2d", getattr(a, "data", a))
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Atleast2d")().infer_shape(a)
    return _emit_shape_node("Atleast2d", [a], {}, out_shape, getattr(a, "dtype", "float32"))


def atleast_3d(a: object) -> Tensor:
    """Atleast 3d."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Atleast3d", getattr(a, "data", a))
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Atleast3d")().infer_shape(a)
    return _emit_shape_node("Atleast3d", [a], {}, out_shape, getattr(a, "dtype", "float32"))


def _get_squeeze_shape(a: object, axis: object) -> tuple:
    shape = list(a.shape) if hasattr(a, "shape") else []
    if axis is None:
        return tuple(s for s in shape if s != 1)
    axes = [axis] if isinstance(axis, int) else axis
    n = {ax + len(shape) if ax < 0 else ax for ax in axes}
    return tuple(shape[i] for i in range(len(shape)) if i not in n)


def squeeze(a: object, axis: object = None) -> Tensor:
    """Squeeze dimensions of a tensor."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Squeeze", getattr(a, "data", a), axis=axis)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    shape_val = _get_squeeze_shape(a, axis)
    return _emit_shape_node("Squeeze", [a], {"axis": axis}, shape_val, getattr(a, "dtype", "float32"))


def diagflat(v: object, k: int = 0) -> Tensor:
    """Create a two-dimensional array with the flattened input as a diagonal."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Diagflat", getattr(v, "data", v), k=k)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(v, "dtype", "float32"), getattr(v, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op

    out_shape = get_op("Diagflat")().infer_shape(v, k=k)
    return _emit_shape_node("Diagflat", [v], {"k": k}, out_shape, getattr(v, "dtype", "float32"))


def block(arrays: object) -> Tensor:
    """Assemble an nd-array from nested lists of blocks."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Block", arrays)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), "float32", getattr(data, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op

    out_shape = get_op("Block")().infer_shape(arrays)
    return _emit_shape_node("Block", [arrays] if hasattr(arrays, "shape") else arrays, {}, out_shape, "float32")


def delete(arr: object, obj: object, axis: int = None) -> Tensor:
    """Return a new array with sub-arrays along an axis deleted."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Delete", getattr(arr, "data", arr), obj, axis=axis)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(arr, "dtype", "float32"), getattr(arr, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op

    out_shape = get_op("Delete")().infer_shape(arr, obj=obj, axis=axis)
    return _emit_shape_node("Delete", [arr], {"obj": obj, "axis": axis}, out_shape, getattr(arr, "dtype", "float32"))


def diag_indices(n: int, ndim: int = 2) -> tuple[Tensor, ...]:
    """Return the indices to access the main diagonal of an array."""
    from ml_switcheroo_compiler.core.tensor import TensorConfig

    if config.eager_mode:
        data = get_active_backend().execute_op("DiagIndices", n, ndim=ndim)
        return tuple(Tensor(d, TensorConfig(getattr(d, "shape", (n,)), "int64", None)) for d in data)
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("DiagIndices")().infer_shape(n, ndim=ndim)
    node = _emit_shape_node("DiagIndices", [n], {"ndim": ndim}, out_shape, "int64")
    out_tensors = []
    from ml_switcheroo_compiler.tracing import builder

    for i in range(ndim):
        item_node = builder.TracingNodeBuilder.emit_tracing_node("GetItem", node, output_index=i, key=str(i))
        item_node._shape = (n,)
        item_node.config = TensorConfig((n,), "int64", None)
        out_tensors.append(item_node)
    return tuple(out_tensors)


def diag_indices_from(arr: object) -> tuple[Tensor, ...]:
    """Return the indices to access the main diagonal of an n-dimensional array."""
    from ml_switcheroo_compiler.core.tensor import TensorConfig

    if config.eager_mode:
        data = get_active_backend().execute_op("DiagIndicesFrom", getattr(arr, "data", arr))
        n = arr.shape[0] if hasattr(arr, "shape") and len(arr.shape) > 0 else 1
        return tuple(Tensor(d, TensorConfig(getattr(d, "shape", (n,)), "int64", getattr(arr, "device", None))) for d in data)
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("DiagIndicesFrom")().infer_shape(arr)
    node = _emit_shape_node("DiagIndicesFrom", [arr], {}, out_shape, "int64")
    out_tensors = []
    from ml_switcheroo_compiler.tracing import builder

    n = arr.shape[0] if hasattr(arr, "shape") and len(arr.shape) > 0 else 1
    ndim = len(arr.shape) if hasattr(arr, "shape") else 1
    for i in range(ndim):
        item_node = builder.TracingNodeBuilder.emit_tracing_node("GetItem", node, output_index=i, key=str(i))
        item_node._shape = (n,)
        item_node.config = TensorConfig((n,), "int64", getattr(arr, "device", None))
        out_tensors.append(item_node)
    return tuple(out_tensors)


def size(input: object, **kwargs: object) -> Tensor:
    """Return the number of elements in a tensor."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Size", getattr(input, "data", input), **kwargs)
        return Tensor(data, TensorConfig((), "int32", None))
    from ml_switcheroo_compiler.ops.base import get_op

    out_shape = get_op("Size")().infer_shape(input, **kwargs)
    return _emit_shape_node("Size", [input], kwargs, out_shape, "int32")


def reshape(a: object, newshape: object) -> Tensor:
    """Reshape a tensor."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Reshape", getattr(a, "data", a), newshape=newshape)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Reshape")().infer_shape(a, newshape=newshape)
    return _emit_shape_node("Reshape", [a], {"newshape": newshape}, out_shape, getattr(a, "dtype", "float32"))


def flip(m: object, axis: object = None) -> Tensor:
    """Reverse the order of elements in an array along the given axis."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Flip", getattr(m, "data", m), axis=axis)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(m, "dtype", "float32"), getattr(m, "device", None)))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Flip", [m], {"axis": axis}, getattr(m, "shape", ()), getattr(m, "dtype", "float32"))


def fliplr(m: object) -> Tensor:
    """Flip array in the left/right direction."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Fliplr", getattr(m, "data", m))
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(m, "dtype", "float32"), getattr(m, "device", None)))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Fliplr", [m], {}, getattr(m, "shape", ()), getattr(m, "dtype", "float32"))


def flipud(m: object) -> Tensor:
    """Flip array in the up/down direction."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Flipud", getattr(m, "data", m))
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(m, "dtype", "float32"), getattr(m, "device", None)))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Flipud", [m], {}, getattr(m, "shape", ()), getattr(m, "dtype", "float32"))
