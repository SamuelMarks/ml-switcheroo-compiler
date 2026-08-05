"""Shape operations for Tensor objects."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config

# pylint: disable=duplicate-code
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.ops.creation.frontend import asarray
from ml_switcheroo_compiler.ops.shape.frontend import reshape
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def concatenate(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    """Concatenate a sequence of tensors along a specified dimension.

    Args:
        tensors (object): The tensors parameter.
        dim (int): The dim parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Concatenate", [getattr(t, "data", t) for t in tensors], axis=dim)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", ()),
                getattr(tensors[0], "dtype", "float32"),
                getattr(tensors[0], "device", None),
            ),
        )
    inputs = list(tensors)
    # shape calculation placeholder
    out_shape = tuple(sum(t.shape[i] for t in tensors) if i == dim else tensors[0].shape[i] for i in range(len(tensors[0].shape)))
    return _emit_shape_node(
        "Concatenate",
        inputs,
        {"axis": dim},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    """Stack a sequence of tensors along a new dimension.

    Args:
        tensors (object): The tensors parameter.
        dim (int): The dim parameter.

    Returns:
        Tensor: Result.
    """
    out_dtype = getattr(tensors[0], "dtype", "float32")
    if hasattr(out_dtype, "name"):
        out_dtype_str = str(out_dtype.name)
    elif hasattr(out_dtype, "__name__"):
        out_dtype_str = str(out_dtype.__name__)
    elif hasattr(out_dtype, "value"):
        out_dtype_str = str(out_dtype.value)
    else:
        out_dtype_str = str(out_dtype).split(".")[-1].lower()

    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Stack", [getattr(t, "data", t) for t in tensors], axis=dim)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", ()),
                out_dtype_str,
                getattr(tensors[0], "device", None),
            ),
        )
    inputs = list(tensors)
    # shape calculation placeholder
    out_shape = inputs[0].shape

    try:
        dt = DType(out_dtype_str)
    except ValueError:
        dt = DType.Float32

    return _emit_shape_node(
        "Stack",
        inputs,
        {"axis": dim},
        out_shape,
        dt,
    )


def vstack(tup: Sequence[Tensor]) -> Tensor:
    """Stack arrays in sequence vertically (row wise).

    Args:
        tup (Sequence[Tensor]): The sequence of tensors to stack

    Returns:
        Tensor: The stacked tensor
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Vstack", [t.data for t in tup])
        return Tensor(data, TensorConfig(data.shape, tup[0].dtype, tup[0].device))
    inputs = list(tup)
    out_shape = get_op("Vstack").infer_shape([t.shape for t in inputs])
    return _emit_shape_node("Vstack", inputs, {}, out_shape, inputs[0].dtype)


def hstack(tup: Sequence[Tensor]) -> Tensor:
    """Stack arrays in sequence horizontally (column wise).

    Args:
        tup (Sequence[Tensor]): The sequence of tensors to stack

    Returns:
        Tensor: The stacked tensor
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Hstack", [t.data for t in tup])
        return Tensor(data, TensorConfig(data.shape, tup[0].dtype, tup[0].device))
    inputs = list(tup)
    out_shape = get_op("Hstack").infer_shape([t.shape for t in inputs])
    return _emit_shape_node("Hstack", inputs, {}, out_shape, inputs[0].dtype)


def dstack(tup: Sequence[Tensor]) -> Tensor:
    """Stack arrays in sequence depth wise (along third axis).

    Args:
        tup (Sequence[Tensor]): The sequence of tensors to stack

    Returns:
        Tensor: The stacked tensor
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Dstack", [t.data for t in tup])
        return Tensor(data, TensorConfig(data.shape, tup[0].dtype, tup[0].device))
    inputs = list(tup)
    out_shape = get_op("Dstack").infer_shape([t.shape for t in inputs])
    return _emit_shape_node("Dstack", inputs, {}, out_shape, inputs[0].dtype)


def append(arr: object, values: object, axis: int | None = None) -> Tensor:
    """Append values to the end of an array.

    Args:
        arr (object): Values are appended to a copy of this array.
        values (object): These values are appended to a copy of arr.
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
        return Tensor(data, TensorConfig(data.shape, arr_dtype, arr_device))

    arr_t = asarray(arr)
    values_t = asarray(values)

    if axis is None:
        arr_t = reshape(arr_t, (-1,))
        values_t = reshape(values_t, (-1,))
        axis = 0

    return concatenate([arr_t, values_t], dim=axis)


def column_stack(tup: Sequence[object]) -> Tensor:
    """Stack 1-D arrays as columns into a 2-D array.

    Args:
        tup (Sequence[object]): Sequence of 1-D or 2-D arrays.

    Returns:
        Tensor: The array formed by stacking the given arrays.
    """
    tensors = [asarray(t) for t in tup]
    arrays = []
    for a in tensors:
        if len(a.shape) < MAGIC_VAL_2:
            arrays.append(reshape(a, (a.shape[0], 1)))
        else:
            arrays.append(a)
    return hstack(arrays)
