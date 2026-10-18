"""Constants & Creation Operations."""

from __future__ import annotations

import typing
from collections.abc import Sequence
from typing import cast

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

from .frontend_utils import _emit_creation_node


def _unpack_shape(shape: Sequence[int | Tensor]) -> tuple[int, ...]:
    """Compute the unpack shape tensor operation.

    Args:
        shape (tuple): The shape parameter.

    Returns:
        tuple: Result.
    """
    unpacked_shape = []
    for s in shape:
        if hasattr(s, "data"):
            s_val = s.data
            if hasattr(s_val, "item"):
                s_val = getattr(s_val, "item")()
            unpacked_shape.append(s_val)
        elif hasattr(s, "item"):
            unpacked_shape.append(s.item())
        else:
            unpacked_shape.append(s)
    return tuple(getattr(unpacked_shape, "__iter__")())


def _infer_dtype(val_arr: TensorOrScalar) -> DType:
    """Infers the DType from a backend array.

    Args:
        val_arr (TensorOrScalar): The val_arr parameter.

    Returns:
        DType: Result.
    """
    dtype_str = str(getattr(val_arr, "dtype", "float32"))
    if dtype_str.startswith("<U") or dtype_str.startswith("|S"):
        return DType.String
    if dtype_str == "Any" or dtype_str == "O":
        return DType.Object
    return DType(dtype_str)


def _get_dtype_val(dtype: DType | str | type | None) -> str | type | None:
    """Get the backend dtype value.

    Args:
        dtype (DType | str | type | None): The dtype parameter.

    Returns:
        str | type | None: Result.
    """
    if hasattr(dtype, "value"):
        return str(getattr(dtype, "value"))
    if hasattr(dtype, "name"):
        return str(dtype)
    return dtype


def _try_create_array(
    backend: TensorOrScalar,
    obj: TensorOrScalar | Tensor | Sequence[TensorOrScalar],
    dtype_val: str | type | None = None,
) -> TensorOrScalar:
    """Try create array.

    Args:
        backend (TensorOrScalar): The backend parameter.
        obj (TensorOrScalar | Tensor | Sequence[TensorOrScalar]): The obj parameter.
        dtype_val (str | type | None): The dtype_val parameter.

    Returns:
        TensorOrScalar: Result.
    """
    if dtype_val is None:
        try:
            return cast(TensorOrScalar, getattr(backend, "array")(obj))
        except ValueError:
            return cast(TensorOrScalar, getattr(backend, "array")(obj, dtype="Any"))
    try:
        return cast(TensorOrScalar, getattr(backend, "array")(obj, dtype=dtype_val))
    except TypeError:
        return cast(TensorOrScalar, getattr(backend, "array")(obj))


def _create_backend_array(
    Any: TensorOrScalar | Tensor | Sequence[TensorOrScalar],
    dtype: DType | str | type | None,
) -> TensorOrScalar:
    """Create the backend array.

    Args:
        Any (TensorOrScalar | Tensor | Sequence[TensorOrScalar]): The Any parameter.
        dtype (DType | str | type | None): The dtype parameter.

    Returns:
        TensorOrScalar: Result.
    """
    backend = get_active_backend()
    dtype_val = None
    if dtype is not None:
        dtype_val = _get_dtype_val(dtype)

    res = _try_create_array(backend, Any, dtype_val)

    return res


def array(
    Any: TensorOrScalar,
    dtype: DType | None = None,
) -> Tensor:
    """Create an array.

    Args:
        Any (TensorOrScalar): The Any parameter.
        dtype (DType | None): The dtype parameter.

    Returns:
        Tensor: Result.
    """
    val_arr = _create_backend_array(Any, dtype)
    if dtype is None:
        dtype = _infer_dtype(val_arr)

    shape = tuple(getattr(val_arr, "shape"))

    if config.eager_mode:
        return Tensor(val_arr, TensorConfig(shape, dtype, config.default_device))

    out_id = TracingNodeBuilder.extract_from_constant(val_arr)[0]

    return Tensor(ProxyTensor(typing.cast(str, out_id), shape, dtype.value), TensorConfig(shape, dtype, config.default_device))


def asarray(
    a: TensorOrScalar,
    dtype: DType | None = None,
) -> Tensor:
    """Convert the input to an array.

    Args:
        a (TensorOrScalar): The a parameter.
        dtype (DType | None): The dtype parameter.

    Returns:
        Tensor: Result.
    """
    if isinstance(a, Tensor):
        if dtype is not None and a.dtype != dtype:
            from ml_switcheroo_compiler.ops.unary import cast as unary_cast

            return unary_cast(a, dtype)
        return a
    return array(a, dtype=dtype)


def convert_to_tensor(
    x: TensorOrScalar,
    dtype: DType | None = None,
) -> Tensor:
    """Convert the given Any to a Tensor.

    Args:
        x (TensorOrScalar): Object to convert.
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
    return cast(Tensor, _emit_creation_node("Zeros", shape, dtype, {}))


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
    return cast(Tensor, _emit_creation_node("Ones", shape, dtype, {}))


def _extract_fill_value(fill_value: float | int | bool | Tensor) -> float | int | bool:
    """Extract fill value.

    Args:
        fill_value (float | int | bool | Tensor): The fill_value parameter.

    Returns:
        float | int | bool: Result.
    """
    val = fill_value.data if hasattr(fill_value, "data") else fill_value
    if hasattr(val, "item"):
        val = getattr(val, "item")()
    import typing

    return typing.cast("float | int | bool", val)


def _full_eager(shape: tuple[int, ...], fill_value: TensorOrScalar, dtype: DType, device: Device) -> Tensor:
    """Full eager.

    Args:
        shape (tuple): The shape parameter.
        fill_value (TensorOrScalar): The fill_value parameter.
        dtype (DType): The dtype parameter.
        device (Device): The device parameter.

    Returns:
        Tensor: Result.
    """
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
        shape (int | Sequence[int]): The shape parameter.
        fill_value (float): The fill_value parameter.
        dtype (DType | None): The dtype parameter.
        device (Device | None): The device parameter.

    Returns:
        Tensor: Result.
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    shape = _unpack_shape((shape,) if isinstance(shape, int) else tuple(shape))
    fill_value = _extract_fill_value(fill_value)

    if config.eager_mode:
        return _full_eager(shape, fill_value, dtype, device)

    return cast(
        Tensor,
        _emit_creation_node(
            "Full",
            shape,
            dtype,
            {"fill_value": fill_value},
        ),
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
    return cast(Tensor, _emit_creation_node("ConstantOfShape", input.shape, dtype, {"value": 0}))


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
    return cast(Tensor, _emit_creation_node("ConstantOfShape", input.shape, dtype, {"value": 1}))


def full_like(
    input: Tensor,
    fill_value: float,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a tensor filled with `fill_value`, with the same size as `input`.

    Args:
        input (Tensor): The input parameter.
        fill_value (float): The fill_value parameter.
        dtype (DType | None): The dtype parameter.
        device (Device | None): The device parameter.

    Returns:
        Tensor: Result.
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
    return cast(
        Tensor,
        _emit_creation_node(
            "ConstantOfShape",
            input.shape,
            dtype,
            {"value": fill_value},
        ),
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
    return cast(Tensor, _emit_creation_node("Zeros", shape, dtype, {}))


def empty_like(x: Tensor, dtype: DType | None = None) -> Tensor:
    """Return a new array with the same shape and type as a given array.

    Args:
        x (Tensor): The input x tensor.
        dtype (DType | None): The target data type.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    return empty(x.shape, dtype=dtype if dtype is not None else x.dtype)


def convert_to_numpy(x: Tensor) -> TensorOrScalar:
    """Convert a tensor to a numpy array.

    Args:
        x (Tensor): Input tensor.

    Returns:
        TensorOrScalar: A numpy array.
    """
    if hasattr(x, "numpy"):
        return x.numpy()
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    return get_active_backend().asarray(getattr(x, "data", x))


def frombuffer(
    buffer: TensorOrScalar,
    dtype: DType | None = None,
    count: int = -1,
    offset: int = 0,
) -> Tensor:
    """Create a 1-D tensor from a buffer.

    Args:
        buffer (TensorOrScalar): An Any that exposes the buffer interface.
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
        shape = getattr(data, "shape") if hasattr(data, "shape") else ()
        return Tensor(data, TensorConfig(shape, dtype, config.default_device))
    return cast(Tensor, _emit_creation_node("Frombuffer", (count,) if count != -1 else (), dtype, {"offset": offset}))
