# pylint: disable=duplicate-code

"""Define shape, memory, and movement operations for Tensor Anys.

This module provides functions to manipulate tensor shapes, dimensions, and memory
layouts, supporting both eager execution (using NumPy) and lazy execution (by tracing
and emitting logical nodes to a graph)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, Union, cast

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, EagerValue, get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def expand_dims(a: EagerValue, axis: int) -> EagerValue:
    """Expand dimensions.

    Args:
        a (Any): The a parameter.
        axis (int): The axis parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("ExpandDims", getattr(a, "data", a), axis=axis)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("ExpandDims")().infer_shape(a, axis=axis)
    return _emit_shape_node("ExpandDims", [a], {"axis": axis}, cast("Sequence[Union[int, str]]", out_shape), getattr(a, "dtype", "float32"))


def argwhere(a: EagerValue) -> EagerValue:
    """Find the indices of array elements that are non-zero, grouped by element.

    Args:
        a (Any): Input data.

    Returns:
        Tensor: Indices of elements that are non-zero.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Argwhere", getattr(a, "data", a))
        return Tensor(data, TensorConfig(getattr(data, "shape"), "int64", getattr(a, "device", None)))
    return _emit_shape_node("Argwhere", [a], {}, cast("Sequence[Union[int, str]]", (None, None)), "int64")


def argpartition(a: EagerValue, kth: EagerValue, axis: int = -1, kind: str = "introselect", order: EagerValue = None) -> EagerValue:
    """Perform an indirect partition along the given axis.

    Args:
        a (Any): The a parameter.
        kth (Any): The kth parameter.
        axis (int): The axis parameter.
        kind (str): The kind parameter.
        order (Any): The order parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Argpartition", getattr(a, "data", a), kth, axis=axis, kind=kind, order=order)
        return Tensor(data, TensorConfig(getattr(data, "shape"), "int64", getattr(a, "device", None)))
    return _emit_shape_node(
        "Argpartition",
        [a, kth],
        {"axis": axis, "kind": kind, "order": order},
        getattr(a, "shape", ()),
        "int64",
    )


def partition(a: EagerValue, kth: EagerValue, axis: int = -1, kind: str = "introselect", order: EagerValue = None) -> EagerValue:
    """Return a partitioned copy of an array.

    Args:
        a (Any): The a parameter.
        kth (Any): The kth parameter.
        axis (int): The axis parameter.
        kind (str): The kind parameter.
        order (Any): The order parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Partition", getattr(a, "data", a), kth, axis=axis, kind=kind, order=order)
        return Tensor(
            data,
            TensorConfig(getattr(data, "shape"), getattr(a, "dtype", "float32"), getattr(a, "device", None)),
        )

    return _emit_shape_node(
        "Partition",
        [a, kth],
        {"axis": axis, "kind": kind, "order": order},
        getattr(a, "shape", ()),
        getattr(a, "dtype", "float32"),
    )


def compress(condition: EagerValue, a: EagerValue, axis: EagerValue = None, out: EagerValue = None) -> EagerValue:
    """Return selected slices of an array along given axis.

    Args:
        condition (Any): The condition parameter.
        a (Any): The a parameter.
        axis (int): The axis parameter.
        out (Any): The out parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Compress", condition, getattr(a, "data", a), axis=axis, out=out)
        return Tensor(
            data,
            TensorConfig(getattr(data, "shape"), getattr(a, "dtype", "float32"), getattr(a, "device", None)),
        )

    return _emit_shape_node(
        "Compress",
        [condition, a],
        {"axis": axis, "out": out},
        cast("Sequence[Union[int, str]]", (None,)),
        getattr(a, "dtype", "float32"),
    )


def insert(arr: EagerValue, obj: EagerValue, values: EagerValue, axis: EagerValue = None) -> EagerValue:
    """Insert values along the given axis before the given indices.

    Args:
        arr (Any): The arr parameter.
        obj (Any): The obj parameter.
        values (Any): The values parameter.
        axis (int): The axis parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Insert", getattr(arr, "data", arr), obj, getattr(values, "data", values), axis=axis)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(arr, "dtype", "float32"), getattr(arr, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op

    out_shape = get_op("Insert")().infer_shape(arr, obj=obj, values=values, axis=axis)
    return _emit_shape_node("Insert", [arr, values] if hasattr(values, "shape") else [arr], {"obj": obj, "axis": axis}, cast("Sequence[Union[int, str]]", out_shape), getattr(arr, "dtype", "float32"))


def fill_diagonal(a: EagerValue, val: EagerValue, wrap: bool = False) -> EagerValue:
    """Fill the main diagonal of the given array of any dimensionality.

    Args:
        a (Any): The a parameter.
        val (Any): The val parameter.
        wrap (bool): The wrap parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("FillDiagonal", getattr(a, "data", a), getattr(val, "data", val), wrap=wrap)
        return Tensor(data, TensorConfig(getattr(a, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op

    out_shape = get_op("FillDiagonal")().infer_shape(a, val=val, wrap=wrap)
    return _emit_shape_node("FillDiagonal", [a, val] if hasattr(val, "shape") else [a], {"val": val, "wrap": wrap}, cast("Sequence[Union[int, str]]", out_shape), getattr(a, "dtype", "float32"))


def moveaxis(a: EagerValue, source: EagerValue, destination: EagerValue) -> EagerValue:
    """Move axes of a tensor.

    Args:
        a (Any): The a parameter.
        source (Any): The source parameter.
        destination (Any): The destination parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Moveaxis", getattr(a, "data", a), source=source, destination=destination)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Moveaxis")().infer_shape(a, source=source, destination=destination)
    return _emit_shape_node("Moveaxis", [a], {"source": source, "destination": destination}, cast("Sequence[Union[int, str]]", out_shape), getattr(a, "dtype", "float32"))


def permute(a: Tensor, dims: tuple[int, ...] | list[int] | None = None) -> Tensor:
    """Permute dimensions of a tensor.

    Args:
        a (Any): The a parameter.
        dims (Any): The dims parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Permute", getattr(a, "data", a), dims=dims)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Permute")().infer_shape(a, dims=dims)
    return _emit_shape_node("Permute", [a], {"dims": dims}, cast("Sequence[Union[int, str]]", out_shape), getattr(a, "dtype", "float32"))


def swapaxes(a: EagerValue, axis1: int, axis2: int) -> EagerValue:
    """Interchange two axes of an array.

    Args:
        a (Any): The a parameter.
        axis1 (int): The axis1 parameter.
        axis2 (int): The axis2 parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Swapaxes", getattr(a, "data", a), axis1=axis1, axis2=axis2)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Swapaxes")().infer_shape(a, axis1=axis1, axis2=axis2)
    return _emit_shape_node("Swapaxes", [a], {"axis1": axis1, "axis2": axis2}, cast("Sequence[Union[int, str]]", out_shape), getattr(a, "dtype", "float32"))


def roll(a: EagerValue, shift: EagerValue, axis: EagerValue = None) -> EagerValue:
    """Roll array elements along a given axis.

    Args:
        a (Any): The a parameter.
        shift (Any): The shift parameter.
        axis (Any): The axis parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Roll", getattr(a, "data", a), shift=shift, axis=axis)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Roll")().infer_shape(a, shift=shift, axis=axis)
    return _emit_shape_node("Roll", [a], {"shift": shift, "axis": axis}, cast("Sequence[Union[int, str]]", out_shape), getattr(a, "dtype", "float32"))


def atleast_1d(a: EagerValue) -> EagerValue:
    """Atleast 1d.

    Args:
        a (Any): The a parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Atleast1d", getattr(a, "data", a))
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Atleast1d")().infer_shape(a)
    return _emit_shape_node("Atleast1d", [a], {}, cast("Sequence[Union[int, str]]", out_shape), getattr(a, "dtype", "float32"))


def atleast_2d(a: EagerValue) -> EagerValue:
    """Atleast 2d.

    Args:
        a (Any): The a parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Atleast2d", getattr(a, "data", a))
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Atleast2d")().infer_shape(a)
    return _emit_shape_node("Atleast2d", [a], {}, cast("Sequence[Union[int, str]]", out_shape), getattr(a, "dtype", "float32"))


def atleast_3d(a: EagerValue) -> EagerValue:
    """Atleast 3d.

    Args:
        a (Any): The a parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Atleast3d", getattr(a, "data", a))
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Atleast3d")().infer_shape(a)
    return _emit_shape_node("Atleast3d", [a], {}, cast("Sequence[Union[int, str]]", out_shape), getattr(a, "dtype", "float32"))


def _get_squeeze_shape(a: Tensor, axis: int | tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    """Evaluate _get_squeeze_shape operation.

    Args:
        a (Any): The a parameter.
        axis (Any): The axis parameter.

    Returns:
        tuple: Result.
    """
    shape = list(a.shape) if hasattr(a, "shape") else []
    if axis is None:
        return tuple(s for s in shape if s != 1)
    axes = [axis] if isinstance(axis, int) else axis
    n = {ax + len(shape) if ax < 0 else ax for ax in axes}
    return tuple(shape[i] for i in range(len(shape)) if i not in n)


def squeeze(a: EagerValue, axis: EagerValue = None) -> EagerValue:
    """Squeeze dimensions of a tensor.

    Args:
        a (Any): The a parameter.
        axis (Any): The axis parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Squeeze", getattr(a, "data", a), axis=axis)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    shape_val = cast("Callable[..., Sequence[Union[int, str]]]", _get_squeeze_shape)(cast("Tensor", a), axis)
    return _emit_shape_node("Squeeze", [a], {"axis": axis}, shape_val, getattr(a, "dtype", "float32"))


def diagflat(v: EagerValue, k: int = 0) -> EagerValue:
    """Create a two-dimensional array with the flattened input as a diagonal.

    Args:
        v (Any): The v parameter.
        k (int): The k parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Diagflat", getattr(v, "data", v), k=k)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(v, "dtype", "float32"), getattr(v, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op

    out_shape = get_op("Diagflat")().infer_shape(v, k=k)
    return _emit_shape_node("Diagflat", [v], {"k": k}, cast("Sequence[Union[int, str]]", out_shape), getattr(v, "dtype", "float32"))


def block(arrays: EagerValue) -> EagerValue:
    """Assemble an nd-array from nested lists of blocks.

    Args:
        arrays (Any): The arrays parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Block", arrays)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), "float32", getattr(data, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op

    out_shape = get_op("Block")().infer_shape(arrays)
    return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", _emit_shape_node)("Block", [arrays] if hasattr(arrays, "shape") else arrays, {}, cast("Sequence[Union[int, str]]", out_shape), "float32"))


def delete(arr: EagerValue, obj: EagerValue, axis: EagerValue = None) -> EagerValue:
    """Return a new array with sub-arrays along an axis deleted.

    Args:
        arr (Any): The arr parameter.
        obj (Any): The obj parameter.
        axis (int): The axis parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Delete", getattr(arr, "data", arr), obj, axis=axis)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(arr, "dtype", "float32"), getattr(arr, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op

    out_shape = get_op("Delete")().infer_shape(arr, obj=obj, axis=axis)
    return _emit_shape_node("Delete", [arr], {"obj": obj, "axis": axis}, cast("Sequence[Union[int, str]]", out_shape), getattr(arr, "dtype", "float32"))


def diag_indices(n: int, ndim: int = 2) -> EagerValue:
    """Return the indices to access the main diagonal of an array.

    Args:
        n (int): The n parameter.
        ndim (int): The ndim parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.tensor import TensorConfig

    if config.eager_mode:
        data = get_active_backend().execute_op("DiagIndices", n, ndim=ndim)
        return tuple(Tensor(d, TensorConfig(getattr(d, "shape", (n,)), "int64", None)) for d in getattr(data, "__iter__")())
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("DiagIndices")().infer_shape(n, ndim=ndim)
    t_n = Tensor(n, TensorConfig((), "int64", None))
    node = _emit_shape_node("DiagIndices", [t_n], {"ndim": ndim}, cast("Sequence[Union[int, str]]", out_shape), "int64")
    out_tensors = []
    from ml_switcheroo_compiler.tracing import builder

    for i in range(ndim):
        item_node = builder.TracingNodeBuilder.emit_tracing_node("GetItem", node, output_index=i, key=str(i))
        item_node._shape = (n,)
        item_node.config = TensorConfig((n,), "int64", None)
        out_tensors.append(item_node)
    return tuple(out_tensors)


def diag_indices_from(arr: EagerValue) -> EagerValue:
    """Return the indices to access the main diagonal of an n-dimensional array.

    Args:
        arr (Any): The arr parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.tensor import TensorConfig

    if config.eager_mode:
        data = get_active_backend().execute_op("DiagIndicesFrom", getattr(arr, "data", arr))
        n = getattr(arr, "shape")[0] if hasattr(arr, "shape") and len(getattr(arr, "shape")) > 0 else 1
        return tuple(Tensor(d, TensorConfig(getattr(d, "shape", (n,)), "int64", getattr(arr, "device", None))) for d in getattr(data, "__iter__")())
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("DiagIndicesFrom")().infer_shape(arr)
    node = _emit_shape_node("DiagIndicesFrom", [arr], {}, cast("Sequence[Union[int, str]]", out_shape), "int64")
    out_tensors = []
    from ml_switcheroo_compiler.tracing import builder

    n = getattr(arr, "shape")[0] if hasattr(arr, "shape") and len(getattr(arr, "shape")) > 0 else 1
    ndim = len(getattr(arr, "shape")) if hasattr(arr, "shape") else 1
    for i in range(ndim):
        item_node = builder.TracingNodeBuilder.emit_tracing_node("GetItem", node, output_index=i, key=str(i))
        item_node._shape = (n,)
        item_node.config = TensorConfig((n,), "int64", getattr(arr, "device", None))
        out_tensors.append(item_node)
    return tuple(out_tensors)


def size(input: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Return the number of elements in a tensor.

    Args:
        input (Any): The input parameter.
        **kwargs (Any): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Size", getattr(input, "data", input), **kwargs)
        return Tensor(data, TensorConfig((), "int32", None))
    from ml_switcheroo_compiler.ops.base import get_op

    out_shape = get_op("Size")().infer_shape(input, **kwargs)
    return _emit_shape_node("Size", [input], kwargs, cast("Sequence[Union[int, str]]", out_shape), "int32")


def reshape(a: Tensor, newshape: int | tuple[int, ...] | list[int]) -> Tensor:
    """Reshape a tensor.

    Args:
        a (Any): The a parameter.
        newshape (Any): The newshape parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Reshape", getattr(a, "data", a), newshape=newshape)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Reshape")().infer_shape(a, newshape=newshape)
    return _emit_shape_node("Reshape", [a], {"newshape": newshape}, cast("Sequence[Union[int, str]]", out_shape), getattr(a, "dtype", "float32"))


def flip(m: Tensor, axis: int | tuple[int, ...] | list[int] | None = None) -> Tensor:
    """Reverse the order of elements in an array along the given axis.

    Args:
        m (Any): The m parameter.
        axis (Any): The axis parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Flip", getattr(m, "data", m), axis=axis)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(m, "dtype", "float32"), getattr(m, "device", None)))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Flip", [m], {"axis": axis}, getattr(m, "shape", ()), getattr(m, "dtype", "float32"))


def fliplr(m: EagerValue) -> EagerValue:
    """Flip array in the left/right direction.

    Args:
        m (Any): The m parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Fliplr", getattr(m, "data", m))
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(m, "dtype", "float32"), getattr(m, "device", None)))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Fliplr", [m], {}, getattr(m, "shape", ()), getattr(m, "dtype", "float32"))


def flipud(m: EagerValue) -> EagerValue:
    """Flip array in the up/down direction.

    Args:
        m (Any): The m parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Flipud", getattr(m, "data", m))
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(m, "dtype", "float32"), getattr(m, "device", None)))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Flipud", [m], {}, getattr(m, "shape", ()), getattr(m, "dtype", "float32"))
