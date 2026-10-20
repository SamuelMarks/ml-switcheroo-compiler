"""Shape operations for Tensor Anys."""

from __future__ import annotations

import typing
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config

# pylint: disable=duplicate-code
from ml_switcheroo_compiler.core.constants import MATRIX_RANK
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.ops.creation.frontend import asarray
from ml_switcheroo_compiler.ops.shape.frontend import reshape
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def concatenate(tensors: Sequence[Tensor], axis: int = 0) -> TensorOrScalar:
    """Concatenate a sequence of tensors along a specified dimension.

    Args:
        tensors (Sequence[Tensor]): The tensors parameter.
        axis (int): The axis parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Concatenate", [getattr(t, "data", t) for t in tensors], axis=axis)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", ()),
                getattr(tensors[0], "dtype", "float32"),
                getattr(tensors[0], "device", None),
            ),
        )
    inputs = list(tensors)
    base_shape = inputs[0].shape if len(inputs) > 0 else ()
    rank = len(base_shape)
    norm_axis = axis if axis >= 0 else axis + rank
    if rank > 0:
        out_shape = tuple(sum(t.shape[norm_axis] for t in tensors) if i == norm_axis else base_shape[i] for i in range(rank))
    else:
        out_shape = ()
    return _emit_shape_node(
        "Concatenate",
        inputs,
        {"axis": axis},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def stack(tensors: Sequence[Tensor], axis: int = 0) -> TensorOrScalar:
    """Stack a sequence of tensors along a new dimension.

    Args:
        tensors (Sequence[Tensor]): The tensors parameter.
        axis (int): The axis parameter.

    Returns:
        Tensor: Result.
    """
    out_dtype = getattr(tensors[0], "dtype", "float32")
    if hasattr(out_dtype, "name"):
        out_dtype_str = str(out_dtype.name).lower()
    elif hasattr(out_dtype, "__name__"):
        out_dtype_str = str(out_dtype.__name__)
    else:
        out_dtype_str = str(getattr(out_dtype, "value", out_dtype)).split(".")[-1].lower()

    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Stack", [getattr(t, "data", t) for t in tensors], axis=axis)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", ()),
                DType(out_dtype_str),
                getattr(tensors[0], "device", None),
            ),
        )
    inputs = list(tensors)
    base_shape = inputs[0].shape if len(inputs) > 0 else ()
    rank = len(base_shape) + 1
    norm_axis = axis if axis >= 0 else axis + rank
    norm_axis = max(0, min(norm_axis, rank - 1))
    out_shape = base_shape[:norm_axis] + (len(inputs),) + base_shape[norm_axis:]

    try:
        dt = DType(DType(out_dtype_str))
    except ValueError:
        dt = DType.Float32

    return _emit_shape_node(
        "Stack",
        inputs,
        {"axis": axis},
        out_shape,
        dt,
    )


def vstack(tup: Sequence[Tensor]) -> TensorOrScalar:
    """Stack arrays in sequence vertically (row wise).

    Args:
        tup (Sequence[Tensor]): The sequence of tensors to stack

    Returns:
        Tensor: The stacked tensor
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Vstack", [t.data for t in tup])
        return Tensor(data, TensorConfig(getattr(data, "shape"), tup[0].dtype, tup[0].device))
    inputs = list(tup)
    out_shape = get_op("Vstack")().infer_shape([t.shape for t in inputs])
    return _emit_shape_node("Vstack", inputs, {}, typing.cast("Sequence[int | str]", out_shape), inputs[0].dtype)


def hstack(tup: Sequence[Tensor]) -> TensorOrScalar:
    """Stack arrays in sequence horizontally (column wise).

    Args:
        tup (Sequence[Tensor]): The sequence of tensors to stack

    Returns:
        Tensor: The stacked tensor
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Hstack", [t.data for t in tup])
        return Tensor(data, TensorConfig(getattr(data, "shape"), tup[0].dtype, tup[0].device))
    inputs = list(tup)
    out_shape = get_op("Hstack")().infer_shape([t.shape for t in inputs])
    return _emit_shape_node("Hstack", inputs, {}, typing.cast("Sequence[int | str]", out_shape), inputs[0].dtype)


def dstack(tup: Sequence[Tensor]) -> TensorOrScalar:
    """Stack arrays in sequence depth wise (along third axis).

    Args:
        tup (Sequence[Tensor]): The sequence of tensors to stack

    Returns:
        Tensor: The stacked tensor
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Dstack", [t.data for t in tup])
        return Tensor(data, TensorConfig(getattr(data, "shape"), tup[0].dtype, tup[0].device))
    inputs = list(tup)
    out_shape = get_op("Dstack")().infer_shape([t.shape for t in inputs])
    return _emit_shape_node("Dstack", inputs, {}, typing.cast("Sequence[int | str]", out_shape), inputs[0].dtype)


def append(arr: TensorOrScalar, values: TensorOrScalar, axis: int | None = None) -> TensorOrScalar:
    """Append values to the end of an array.

    Args:
        arr (TensorOrScalar): Values are appended to a copy of this array.
        values (TensorOrScalar): These values are appended to a copy of arr.
        axis (int | None, optional): The axis along which values are appended.
            If axis is not given, both arr and values are flattened before use.

    Returns:
        Tensor: A copy of arr with values appended to axis.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Append", getattr(arr, "data", arr), getattr(values, "data", values), axis=axis)
        arr_dtype = getattr(arr, "dtype", getattr(values, "dtype", "float32"))
        arr_device = getattr(arr, "device", None)
        return Tensor(data, TensorConfig(getattr(data, "shape"), arr_dtype, arr_device))

    arr_t = asarray(arr)
    values_t = asarray(values)

    if axis is None:
        arr_t = reshape(arr_t, (-1,))
        values_t = reshape(values_t, (-1,))
        axis = 0

    return concatenate([arr_t, values_t], axis=axis)


def column_stack(tup: TensorOrScalar) -> TensorOrScalar:
    """Stack 1-D arrays as columns into a 2-D array.

    Args:
        tup (Sequence[Any]): Sequence of 1-D or 2-D arrays.

    Returns:
        Tensor: The array formed by stacking the given arrays.
    """
    tensors = [asarray(t) for t in getattr(tup, "__iter__")()]
    arrays = []
    for a in tensors:
        if len(a.shape) < MATRIX_RANK:
            arrays.append(reshape(a, (a.shape[0], 1)))
        else:
            arrays.append(a)
    return hstack(arrays)
