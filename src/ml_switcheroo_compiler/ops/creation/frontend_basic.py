"""Constants & Creation Operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.unary import cast
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

from .frontend_utils import _emit_creation_node

if TYPE_CHECKING:
    pass


def _unpack_shape(shape: tuple) -> tuple:
    """Evaluate and process the unpack shape operation.

    Args:
        shape (tuple): Required parameter for shape.

    Returns:
        tuple: The evaluated or processed output.
    """
    unpacked_shape = []
    for s in shape:
        if hasattr(s, "data"):
            s_val = s.data
            if hasattr(s_val, "item"):
                s_val = s_val.item()
            unpacked_shape.append(s_val)
        elif hasattr(s, "item"):
            unpacked_shape.append(s.item())
        else:
            unpacked_shape.append(s)
    return tuple(unpacked_shape)


def _infer_dtype(val_arr: object) -> DType:
    """Infers the DType from a backend array."""
    dtype_str = str(val_arr.dtype)
    if dtype_str.startswith("<U") or dtype_str.startswith("|S"):
        return DType.String
    if dtype_str == "object" or dtype_str == "O":
        return DType.Object
    return DType(dtype_str)


def _get_dtype_val(dtype: object) -> object:
    """Gets the backend dtype value."""
    if hasattr(dtype, "value"):
        return dtype.value
    if hasattr(dtype, "name"):
        return str(dtype)
    return dtype


def _try_create_array(backend: object, obj: object, dtype_val: object = None) -> object:
    """Try create array."""
    if dtype_val is None:
        try:
            return backend.array(obj)
        except ValueError:
            return backend.array(obj, dtype="object")
    try:
        return backend.array(obj, dtype=dtype_val)
    except TypeError:
        return backend.array(obj)


def _create_backend_array(object: object, dtype: object) -> object:
    """Creates the backend array."""
    backend = get_active_backend()
    dtype_val = None
    if dtype is not None:
        dtype_val = _get_dtype_val(dtype)

    res = _try_create_array(backend, object, dtype_val)

    return res


def array(
    object: object,
    dtype: DType | None = None,
) -> Tensor:
    """Creates an array.

    Args:
        object (Any): Argument object to convert
        dtype (Optional[DType]): The data type

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    val_arr = _create_backend_array(object, dtype)
    if dtype is None:
        dtype = _infer_dtype(val_arr)

    shape = tuple(val_arr.shape)

    if config.eager_mode:
        return Tensor(val_arr, TensorConfig(shape, dtype, config.default_device))

    out_id = TracingNodeBuilder.extract_from_constant(val_arr)[0]

    return Tensor(ProxyTensor(out_id, shape, dtype.value), TensorConfig(shape, dtype, config.default_device))


def asarray(
    a: object,
    dtype: DType | None = None,
) -> Tensor:
    """Converts the input to an array.

    Args:
        a (Any): Argument object to convert
        dtype (Optional[DType]): The data type

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    if isinstance(a, Tensor):
        if dtype is not None and a.dtype != dtype:
            return cast(a, dtype)
        return a
    return array(a, dtype=dtype)


def convert_to_tensor(
    x: object,
    dtype: DType | None = None,
) -> Tensor:
    """Converts the given object to a Tensor.

    Args:
        x (object): Object to convert.
        dtype (Optional[DType]): The data type.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    return asarray(x, dtype=dtype)


def zeros(
    shape: int | Sequence[int],
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a tensor filled with the scalar value 0.

    Args:
        shape (Union[int, Sequence[int]]): The shape of the tensor.
        dtype (Optional[DType]): The data type
        device (Optional[Device]): The device to store the tensor on.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    shape = _unpack_shape((shape,) if isinstance(shape, int) else tuple(shape))

    if config.eager_mode:
        data = get_active_backend().execute_op(
            "Zeros",
            shape,
            dtype=dtype.value if hasattr(dtype, "value") else getattr(dtype, "name", str(dtype)),
        )
        return Tensor(data, TensorConfig(shape, dtype, device))
    return _emit_creation_node("Zeros", shape, dtype, {})


def ones(
    shape: int | Sequence[int],
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a tensor filled with the scalar value 1.

    Args:
        shape (Union[int, Sequence[int]]): The shape of the tensor.
        dtype (Optional[DType]): The data type
        device (Optional[Device]): The device to store the tensor on.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    shape = _unpack_shape((shape,) if isinstance(shape, int) else tuple(shape))

    if config.eager_mode:
        data = get_active_backend().execute_op(
            "Ones",
            shape,
            dtype=dtype.value if hasattr(dtype, "value") else getattr(dtype, "name", str(dtype)),
        )
        return Tensor(data, TensorConfig(shape, dtype, device))
    return _emit_creation_node("Ones", shape, dtype, {})


def _extract_fill_value(fill_value: object) -> object:
    """Extract fill value."""
    if hasattr(fill_value, "data"):
        fill_value = fill_value.data
    if hasattr(fill_value, "item"):
        fill_value = fill_value.item()
    return fill_value


def _full_eager(shape: tuple[int, ...], fill_value: object, dtype: DType, device: Device) -> Tensor:
    """Full eager."""
    dt_val = dtype.value if hasattr(dtype, "value") else getattr(dtype, "name", str(dtype))
    data = get_active_backend().execute_op("Full", shape, fill_value, dtype=dt_val)
    return Tensor(data, TensorConfig(shape, dtype, device))


def full(
    shape: int | Sequence[int],
    fill_value: float,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a tensor filled with `fill_value`.

    Args:
        shape (Union[int, Sequence[int]]): The shape of the tensor.
        fill_value (Union[float, int]): The value to fill the tensor with.
        dtype (Optional[DType]): The data type
        device (Optional[Device]): The device to store the tensor on.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    shape = _unpack_shape((shape,) if isinstance(shape, int) else tuple(shape))
    fill_value = _extract_fill_value(fill_value)

    if config.eager_mode:
        return _full_eager(shape, fill_value, dtype, device)

    return _emit_creation_node(
        "Full",
        shape,
        dtype,
        {"fill_value": fill_value},
    )


def zeros_like(
    input: Tensor,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a tensor filled with scalar 0, with the same size as `input`.

    Args:
        input (Tensor): The input tensor
        dtype (Optional[DType]): The data type
        device (Optional[Device]): The device to store the tensor on.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    dtype = dtype or input.dtype
    device = device or input.device
    if config.eager_mode:
        data = get_active_backend().execute_op(
            "Zeros_like",
            input.data,
            dtype=dtype.value if hasattr(dtype, "value") else getattr(dtype, "name", str(dtype)),
        )
        return Tensor(data, TensorConfig(input.shape, dtype, device))
    return _emit_creation_node("ConstantOfShape", input.shape, dtype, {"value": 0})


def ones_like(
    input: Tensor,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a tensor filled with scalar 1, with the same size as `input`.

    Args:
        input (Tensor): The input tensor
        dtype (Optional[DType]): The data type
        device (Optional[Device]): The device to store the tensor on.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    dtype = dtype or input.dtype
    device = device or input.device
    if config.eager_mode:
        data = get_active_backend().execute_op(
            "Ones_like",
            input.data,
            dtype=dtype.value if hasattr(dtype, "value") else getattr(dtype, "name", str(dtype)),
        )
        return Tensor(data, TensorConfig(input.shape, dtype, device))
    return _emit_creation_node("ConstantOfShape", input.shape, dtype, {"value": 1})


def full_like(
    input: Tensor,
    fill_value: float,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a tensor filled with `fill_value`, with the same size as `input`.

    Args:
        input (Tensor): The input tensor
        fill_value (Union[float, int]): The value to fill the tensor with.
        dtype (Optional[DType]): The data type
        device (Optional[Device]): The device to store the tensor on.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    dtype = dtype or input.dtype
    device = device or input.device
    if config.eager_mode:
        data = get_active_backend().execute_op(
            "Full_like",
            input.data,
            fill_value,
            dtype=dtype.value if hasattr(dtype, "value") else getattr(dtype, "name", str(dtype)),
        )
        return Tensor(data, TensorConfig(input.shape, dtype, device))
    return _emit_creation_node(
        "ConstantOfShape",
        input.shape,
        dtype,
        {"value": fill_value},
    )


def empty(
    shape: int | Sequence[int],
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a tensor filled with uninitialized data.

    Args:
        shape (Union[int, Sequence[int]]): The shape of the tensor.
        dtype (Optional[DType]): The data type
        device (Optional[Device]): The device to store the tensor on.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    shape = _unpack_shape((shape,) if isinstance(shape, int) else tuple(shape))

    if config.eager_mode:
        data = get_active_backend().execute_op(
            "Empty",
            shape,
            dtype=dtype.value if hasattr(dtype, "value") else getattr(dtype, "name", str(dtype)),
        )
        return Tensor(data, TensorConfig(shape, dtype, device))
    return _emit_creation_node("Zeros", shape, dtype, {})


def empty_like(x: Tensor, dtype: DType | None = None) -> Tensor:
    """Return a new array with the same shape and type as a given array.

    Args:
        x (Tensor): The input x tensor.
        dtype (DType | None): The target data type.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    return empty(x.shape, dtype=dtype if dtype is not None else x.dtype)


def convert_to_numpy(x: Tensor) -> object:
    """Converts a tensor to a numpy array.

    Args:
        x (Tensor): Input tensor.

    Returns:
        object: A numpy array.
    """
    if hasattr(x, "numpy"):
        return x.numpy()
    import numpy as np

    return np.array(getattr(x, "data", x))


def frombuffer(
    buffer: object,
    dtype: DType | None = None,
    count: int = -1,
    offset: int = 0,
) -> Tensor:
    """Create a 1-D tensor from a buffer.

    Args:
        buffer (object): An object that exposes the buffer interface.
        dtype (Optional[DType]): Data type of the returned array.
        count (int): Number of items to read. -1 means all data in the buffer.
        offset (int): Start reading the buffer from this offset.

    Returns:
        Tensor: A tensor containing the result.
    """
    dtype = dtype or config.default_float_dtype
    if config.eager_mode:
        data = get_active_backend().execute_op(
            "Frombuffer",
            buffer,
            dtype=dtype.value if hasattr(dtype, "value") else getattr(dtype, "name", str(dtype)),
            count=count,
            offset=offset,
        )
        shape = data.shape if hasattr(data, "shape") else ()
        return Tensor(data, TensorConfig(shape, dtype, config.default_device))
    return _emit_creation_node("Frombuffer", (count,) if count != -1 else (), dtype, {"offset": offset})
