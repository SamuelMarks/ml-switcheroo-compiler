"""Constants & Creation Operations."""

from __future__ import annotations


from typing import TYPE_CHECKING


from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType


from .frontend_utils import _emit_creation_node, _emit_constant_node


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
    if dtype is None:
        val_arr = get_active_backend().array(object)
        from ml_switcheroo_compiler.core.dtype import DType

        dtype_str = str(val_arr.dtype)
        if dtype_str.startswith("<U") or dtype_str.startswith("|S") or dtype_str == "object":
            dtype = DType.String
        else:
            dtype = DType(dtype_str)
    else:
        val_arr = get_active_backend().array(object)

    shape = tuple(val_arr.shape)

    if config.eager_mode:
        return Tensor(val_arr, TensorConfig(shape, dtype, config.default_device))

    return _emit_constant_node(object, dtype)


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
            from ml_switcheroo_compiler.ops.unary import cast

            return cast(a, dtype)
        return a
    return array(a, dtype=dtype)


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
    shape = (shape,) if isinstance(shape, int) else tuple(shape)

    if config.eager_mode:
        data = get_active_backend().execute_op("Zeros", shape, dtype=dtype.value)
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
    shape = (shape,) if isinstance(shape, int) else tuple(shape)

    if config.eager_mode:
        data = get_active_backend().execute_op("Ones", shape, dtype=dtype.value)
        return Tensor(data, TensorConfig(shape, dtype, device))
    return _emit_creation_node("Ones", shape, dtype, {})


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
    shape = (shape,) if isinstance(shape, int) else tuple(shape)

    if config.eager_mode:
        data = get_active_backend().execute_op("Full", shape, fill_value, dtype=dtype.value)
        return Tensor(data, TensorConfig(shape, dtype, device))
    return _emit_creation_node(
        "ConstantOfShape",
        shape,
        dtype,
        {"value": fill_value},
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
        data = get_active_backend().execute_op("Zeros_like", input.data, dtype=dtype.value)
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
        data = get_active_backend().execute_op("Ones_like", input.data, dtype=dtype.value)
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
            dtype=dtype.value,
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
    shape = (shape,) if isinstance(shape, int) else tuple(shape)

    if config.eager_mode:
        data = get_active_backend().execute_op("Empty", shape, dtype=dtype.value)
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
