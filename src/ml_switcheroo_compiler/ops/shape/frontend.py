# pylint: disable=duplicate-code

"""Define shape, memory, and movement operations for Tensor objects.

This module provides functions to manipulate tensor shapes, dimensions, and memory
layouts, supporting both eager execution (using NumPy) and lazy execution (by tracing
and emitting logical nodes to a graph)
"""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import (
    ensure_callable,
    ensure_eager_list,
    ensure_eager_tensor,
    ensure_int,
    ensure_shape,
    get_active_backend,
)
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import (
    ShapeProtocol,
    TensorOrScalar,
    extract_data,
    extract_device,
    extract_dtype,
    extract_shape,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig, ensure_tensor
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def expand_dims(a: TensorOrScalar, axis: int) -> TensorOrScalar:
    """Expand dimensions.

    Args:
        a (TensorOrScalar): Input `a` argument (a).
        axis (int): Input `axis` argument (axis).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("ExpandDims", extract_data(a), axis=axis)
        return Tensor(data, TensorConfig(extract_shape(data), extract_dtype(a, "float32"), extract_device(a, None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("ExpandDims")().infer_shape(a, axis=axis)
    return _emit_shape_node("ExpandDims", [a], {"axis": axis}, out_shape, extract_dtype(a, "float32"))


def argwhere(a: TensorOrScalar) -> TensorOrScalar:
    """Find the indices of array elements that are non-zero, grouped by element.

    Args:
        a (TensorOrScalar): Input data.

    Returns:
        Tensor: Indices of elements that are non-zero.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Argwhere", extract_data(a))
        return Tensor(data, TensorConfig(extract_shape(data), "int64", extract_device(a, None)))
    return _emit_shape_node("Argwhere", [a], {}, (None, None), "int64")


def argpartition(a: TensorOrScalar, kth: TensorOrScalar, axis: int = -1, kind: str = "introselect", order: TensorOrScalar = None) -> TensorOrScalar:
    """Perform an indirect partition along the given axis.

    Args:
        a (TensorOrScalar): Input `a` argument (a).
        kth (TensorOrScalar): Input `kth` argument (kth).
        axis (int): Input `axis` argument (axis).
        kind (str): Input `kind` argument (kind).
        order (TensorOrScalar): Input `order` argument (order).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Argpartition", extract_data(a), kth, axis=axis, kind=kind, order=order)
        return Tensor(data, TensorConfig(extract_shape(data), "int64", extract_device(a, None)))
    return _emit_shape_node(
        "Argpartition",
        [a, kth],
        {"axis": axis, "kind": kind, "order": order},
        extract_shape(a),
        "int64",
    )


def partition(a: TensorOrScalar, kth: TensorOrScalar, axis: int = -1, kind: str = "introselect", order: TensorOrScalar = None) -> TensorOrScalar:
    """Return a partitioned copy of an array.

    Args:
        a (TensorOrScalar): Input `a` argument (a).
        kth (TensorOrScalar): Input `kth` argument (kth).
        axis (int): Input `axis` argument (axis).
        kind (str): Input `kind` argument (kind).
        order (TensorOrScalar): Input `order` argument (order).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Partition", extract_data(a), kth, axis=axis, kind=kind, order=order)
        return Tensor(
            data,
            TensorConfig(extract_shape(data), extract_dtype(a, "float32"), extract_device(a, None)),
        )

    return _emit_shape_node(
        "Partition",
        [a, kth],
        {"axis": axis, "kind": kind, "order": order},
        extract_shape(a),
        extract_dtype(a, "float32"),
    )


def compress(condition: TensorOrScalar, a: TensorOrScalar, axis: TensorOrScalar = None, out: TensorOrScalar = None) -> TensorOrScalar:
    """Return selected slices of an array along given axis.

    Args:
        condition (TensorOrScalar): Input `condition` argument (condition).
        a (TensorOrScalar): Input `a` argument (a).
        axis (int): Input `axis` argument (axis).
        out (TensorOrScalar): Input `out` argument (out).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Compress", condition, extract_data(a), axis=axis, out=out)
        return Tensor(
            data,
            TensorConfig(extract_shape(data), extract_dtype(a, "float32"), extract_device(a, None)),
        )

    return _emit_shape_node(
        "Compress",
        [condition, a],
        {"axis": axis, "out": out},
        (None,),
        extract_dtype(a, "float32"),
    )


def insert(arr: TensorOrScalar, obj: TensorOrScalar, values: TensorOrScalar, axis: TensorOrScalar = None) -> TensorOrScalar:
    """Insert values along the given axis before the given indices.

    Args:
        arr (TensorOrScalar): Input `arr` argument (arr).
        obj (TensorOrScalar): Input `obj` argument (obj).
        values (TensorOrScalar): Input `values` argument (values).
        axis (int): Input `axis` argument (axis).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Insert", extract_data(arr), obj, extract_data(values), axis=axis)
        return Tensor(data, TensorConfig(extract_shape(data), extract_dtype(arr, "float32"), extract_device(arr, None)))
    from ml_switcheroo_compiler.ops.base import get_op

    out_shape = get_op("Insert")().infer_shape(arr, obj=obj, values=values, axis=axis)
    return _emit_shape_node("Insert", [arr, values] if isinstance(values, ShapeProtocol) else [arr], {"obj": obj, "axis": axis}, out_shape, extract_dtype(arr, "float32"))


def fill_diagonal(a: TensorOrScalar, val: TensorOrScalar, wrap: bool = False) -> TensorOrScalar:
    """Fill the main diagonal of the given array of any dimensionality.

    Args:
        a (TensorOrScalar): Input `a` argument (a).
        val (TensorOrScalar): Input `val` argument (val).
        wrap (bool): Input `wrap` argument (wrap).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("FillDiagonal", extract_data(a), extract_data(val), wrap=wrap)
        return Tensor(data, TensorConfig(extract_shape(a), extract_dtype(a, "float32"), extract_device(a, None)))
    from ml_switcheroo_compiler.ops.base import get_op

    out_shape = get_op("FillDiagonal")().infer_shape(a, val=val, wrap=wrap)
    return _emit_shape_node("FillDiagonal", [a, val] if isinstance(val, ShapeProtocol) else [a], {"val": val, "wrap": wrap}, out_shape, extract_dtype(a, "float32"))


def moveaxis(a: TensorOrScalar, source: TensorOrScalar, destination: TensorOrScalar) -> TensorOrScalar:
    """Move axes of a tensor.

    Args:
        a (TensorOrScalar): Input `a` argument (a).
        source (TensorOrScalar): Input `source` argument (source).
        destination (TensorOrScalar): Input `destination` argument (destination).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Moveaxis", extract_data(a), source=source, destination=destination)
        return Tensor(data, TensorConfig(extract_shape(data), extract_dtype(a, "float32"), extract_device(a, None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Moveaxis")().infer_shape(a, source=source, destination=destination)
    return _emit_shape_node("Moveaxis", [a], {"source": source, "destination": destination}, out_shape, extract_dtype(a, "float32"))


def permute(a: Tensor, dims: tuple[int, ...] | list[int] | None = None) -> Tensor:
    """Permute dimensions of a tensor.

    Args:
        a (Tensor): Input `a` argument (a).
        dims (tuple[int, ...] | list[int] | None): Input `dims` argument (dims).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Permute", extract_data(a), dims=dims)
        return Tensor(data, TensorConfig(extract_shape(data), extract_dtype(a, "float32"), extract_device(a, None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Permute")().infer_shape(a, dims=dims)
    return _emit_shape_node("Permute", [a], {"dims": dims}, out_shape, extract_dtype(a, "float32"))


def swapaxes(a: TensorOrScalar, axis1: int, axis2: int) -> TensorOrScalar:
    """Interchange two axes of an array.

    Args:
        a (TensorOrScalar): Input `a` argument (a).
        axis1 (int): Input `axis1` argument (axis1).
        axis2 (int): Input `axis2` argument (axis2).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Swapaxes", extract_data(a), axis1=axis1, axis2=axis2)
        return Tensor(data, TensorConfig(extract_shape(data), extract_dtype(a, "float32"), extract_device(a, None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Swapaxes")().infer_shape(a, axis1=axis1, axis2=axis2)
    return _emit_shape_node("Swapaxes", [a], {"axis1": axis1, "axis2": axis2}, out_shape, extract_dtype(a, "float32"))


def roll(a: TensorOrScalar, shift: TensorOrScalar, axis: TensorOrScalar = None) -> TensorOrScalar:
    """Roll array elements along a given axis.

    Args:
        a (TensorOrScalar): Input `a` argument (a).
        shift (TensorOrScalar): Input `shift` argument (shift).
        axis (TensorOrScalar): Input `axis` argument (axis).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Roll", extract_data(a), shift=shift, axis=axis)
        return Tensor(data, TensorConfig(extract_shape(data), extract_dtype(a, "float32"), extract_device(a, None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Roll")().infer_shape(a, shift=shift, axis=axis)
    return _emit_shape_node("Roll", [a], {"shift": shift, "axis": axis}, out_shape, extract_dtype(a, "float32"))


def atleast_1d(a: TensorOrScalar) -> TensorOrScalar:
    """Atleast 1d.

    Args:
        a (TensorOrScalar): Input `a` argument (a).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Atleast1d", extract_data(a))
        return Tensor(data, TensorConfig(extract_shape(data), extract_dtype(a, "float32"), extract_device(a, None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Atleast1d")().infer_shape(a)
    return _emit_shape_node("Atleast1d", [a], {}, out_shape, extract_dtype(a, "float32"))


def atleast_2d(a: TensorOrScalar) -> TensorOrScalar:
    """Atleast 2d.

    Args:
        a (TensorOrScalar): Input `a` argument (a).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Atleast2d", extract_data(a))
        return Tensor(data, TensorConfig(extract_shape(data), extract_dtype(a, "float32"), extract_device(a, None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Atleast2d")().infer_shape(a)
    return _emit_shape_node("Atleast2d", [a], {}, out_shape, extract_dtype(a, "float32"))


def atleast_3d(a: TensorOrScalar) -> TensorOrScalar:
    """Atleast 3d.

    Args:
        a (TensorOrScalar): Input `a` argument (a).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Atleast3d", extract_data(a))
        return Tensor(data, TensorConfig(extract_shape(data), extract_dtype(a, "float32"), extract_device(a, None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Atleast3d")().infer_shape(a)
    return _emit_shape_node("Atleast3d", [a], {}, out_shape, extract_dtype(a, "float32"))


def _get_squeeze_shape(a: TensorOrScalar, axis: TensorOrScalar) -> tuple[int, ...]:
    """Compute the get squeeze shape tensor operation.

    Args:
        a (Tensor): Input `a` argument (a).
        axis (int | tuple[int, ...] | list[int] | None): Input `axis` argument (axis).

    Returns:
        tuple: Computed `tuple` output.
    """
    shape = list(extract_shape(a))
    if axis is None:
        return tuple(s for s in shape if s != 1)
    axes = [axis] if isinstance(axis, int) else ensure_shape(axis)
    n = {ensure_int(ax) + len(shape) if ensure_int(ax) < 0 else ensure_int(ax) for ax in axes}
    return tuple(shape[i] for i in range(len(shape)) if i not in n)


def squeeze(a: TensorOrScalar, axis: TensorOrScalar = None) -> TensorOrScalar:
    """Squeeze dimensions of a tensor.

    Args:
        a (TensorOrScalar): Input `a` argument (a).
        axis (TensorOrScalar): Input `axis` argument (axis).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Squeeze", extract_data(a), axis=axis)
        return Tensor(data, TensorConfig(extract_shape(data), extract_dtype(a, "float32"), extract_device(a, None)))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    shape_val = _get_squeeze_shape(ensure_tensor(a), axis)
    return _emit_shape_node("Squeeze", [a], {"axis": axis}, shape_val, extract_dtype(a, "float32"))


def diagflat(v: TensorOrScalar, k: int = 0) -> TensorOrScalar:
    """Create a two-dimensional array with the flattened input as a diagonal.

    Args:
        v (TensorOrScalar): Input `v` argument (v).
        k (int): Input `k` argument (k).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Diagflat", extract_data(v), k=k)
        return Tensor(data, TensorConfig(extract_shape(data), extract_dtype(v, "float32"), extract_device(v, None)))
    from ml_switcheroo_compiler.ops.base import get_op

    out_shape = get_op("Diagflat")().infer_shape(v, k=k)
    return _emit_shape_node("Diagflat", [v], {"k": k}, out_shape, extract_dtype(v, "float32"))


def block(arrays: TensorOrScalar) -> TensorOrScalar:
    """Assemble an nd-array from nested lists of blocks.

    Args:
        arrays (TensorOrScalar): Input `arrays` argument (arrays).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Block", arrays)
        return Tensor(data, TensorConfig(extract_shape(data), "float32", extract_device(data, None)))
    from ml_switcheroo_compiler.ops.base import get_op

    out_shape = get_op("Block")().infer_shape(arrays)
    return ensure_eager_tensor(ensure_callable(_emit_shape_node)("Block", [arrays] if isinstance(arrays, ShapeProtocol) else arrays, {}, out_shape, "float32"))


def delete(arr: TensorOrScalar, obj: TensorOrScalar, axis: TensorOrScalar = None) -> TensorOrScalar:
    """Return a new array with sub-arrays along an axis deleted.

    Args:
        arr (TensorOrScalar): Input `arr` argument (arr).
        obj (TensorOrScalar): Input `obj` argument (obj).
        axis (int): Input `axis` argument (axis).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Delete", extract_data(arr), obj, axis=axis)
        return Tensor(data, TensorConfig(extract_shape(data), extract_dtype(arr, "float32"), extract_device(arr, None)))
    from ml_switcheroo_compiler.ops.base import get_op

    out_shape = get_op("Delete")().infer_shape(arr, obj=obj, axis=axis)
    return _emit_shape_node("Delete", [arr], {"obj": obj, "axis": axis}, out_shape, extract_dtype(arr, "float32"))


def diag_indices(n: int, ndim: int = 2) -> TensorOrScalar:
    """Return the indices to access the main diagonal of an array.

    Args:
        n (int): Input `n` argument (n).
        ndim (int): Input `ndim` argument (ndim).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.core.tensor import TensorConfig

    if config.eager_mode:
        data = get_active_backend().execute_op("DiagIndices", n, ndim=ndim)
        return tuple(Tensor(d, TensorConfig(extract_shape(d, (n,)), "int64", None)) for d in ensure_eager_list(data))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("DiagIndices")().infer_shape(n, ndim=ndim)
    t_n = Tensor(n, TensorConfig((), "int64", None))
    node = _emit_shape_node("DiagIndices", [t_n], {"ndim": ndim}, out_shape, "int64")
    out_tensors = []
    from ml_switcheroo_compiler.tracing import builder

    for i in range(ndim):
        item_node = builder.TracingNodeBuilder.emit_tracing_node("GetItem", node, output_index=i, key=str(i))
        item_node._shape = (n,)
        item_node.config = TensorConfig((n,), "int64", None)
        out_tensors.append(item_node)
    return tuple(out_tensors)


def diag_indices_from(arr: TensorOrScalar) -> TensorOrScalar:
    """Return the indices to access the main diagonal of an n-dimensional array.

    Args:
        arr (TensorOrScalar): Input `arr` argument (arr).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.core.tensor import TensorConfig

    arr_shape = extract_shape(arr)
    if config.eager_mode:
        data = get_active_backend().execute_op("DiagIndicesFrom", extract_data(arr))
        n = arr_shape[0] if len(arr_shape) > 0 else 1
        return tuple(Tensor(d, TensorConfig(extract_shape(d, (n,)), "int64", extract_device(arr, None))) for d in ensure_eager_list(data))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("DiagIndicesFrom")().infer_shape(arr)
    node = _emit_shape_node("DiagIndicesFrom", [arr], {}, out_shape, "int64")
    out_tensors = []
    from ml_switcheroo_compiler.tracing import builder

    n = arr_shape[0] if len(arr_shape) > 0 else 1
    ndim = len(arr_shape) if isinstance(arr, ShapeProtocol) else 1
    for i in range(ndim):
        item_node = builder.TracingNodeBuilder.emit_tracing_node("GetItem", node, output_index=i, key=str(i))
        item_node._shape = (n,)
        item_node.config = TensorConfig((n,), "int64", extract_device(arr, None))
        out_tensors.append(item_node)
    return tuple(out_tensors)


def size(input: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Return the number of elements in a tensor.

    Args:
        input (TensorOrScalar): Input `input` argument (input).
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Size", extract_data(input), **kwargs)
        return Tensor(data, TensorConfig((), "int32", None))
    from ml_switcheroo_compiler.ops.base import get_op

    out_shape = get_op("Size")().infer_shape(input, **kwargs)
    return _emit_shape_node("Size", [input], kwargs, out_shape, "int32")


def reshape(a: Tensor, newshape: int | tuple[int, ...] | list[int]) -> Tensor:
    """Reshape a tensor.

    Args:
        a (Tensor): Input `a` argument (a).
        newshape (int | tuple[int, ...] | list[int]): Input `newshape` argument (newshape).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Reshape", extract_data(a), newshape=newshape)
        return Tensor(data, TensorConfig(extract_shape(data), extract_dtype(a, "float32"), extract_device(a, None)))
    from ml_switcheroo_compiler.ops.base import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out_shape = get_op("Reshape")().infer_shape(a, newshape=newshape)
    return _emit_shape_node("Reshape", [a], {"newshape": newshape}, out_shape, extract_dtype(a, "float32"))


def flip(m: Tensor, axis: int | tuple[int, ...] | list[int] | None = None) -> Tensor:
    """Reverse the order of elements in an array along the given axis.

    Args:
        m (Tensor): Input `m` argument (m).
        axis (int | tuple[int, ...] | list[int] | None): Input `axis` argument (axis).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Flip", extract_data(m), axis=axis)
        return Tensor(data, TensorConfig(extract_shape(data), extract_dtype(m, "float32"), extract_device(m, None)))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Flip", [m], {"axis": axis}, extract_shape(m), extract_dtype(m, "float32"))


def fliplr(m: TensorOrScalar) -> TensorOrScalar:
    """Flip array in the left/right direction.

    Args:
        m (TensorOrScalar): Input `m` argument (m).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Fliplr", extract_data(m))
        return Tensor(data, TensorConfig(extract_shape(data), extract_dtype(m, "float32"), extract_device(m, None)))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Fliplr", [m], {}, extract_shape(m), extract_dtype(m, "float32"))


def flipud(m: TensorOrScalar) -> TensorOrScalar:
    """Flip array in the up/down direction.

    Args:
        m (TensorOrScalar): Input `m` argument (m).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Flipud", extract_data(m))
        return Tensor(data, TensorConfig(extract_shape(data), extract_dtype(m, "float32"), extract_device(m, None)))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Flipud", [m], {}, extract_shape(m), extract_dtype(m, "float32"))
