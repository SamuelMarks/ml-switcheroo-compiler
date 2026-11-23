"""Constants & Creation Operations."""

from __future__ import annotations

import typing
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import coerce_int, coerce_str, get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    DataHolderProtocol,
    OpValue,
    extract_data,
    extract_dtype,
    extract_shape,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig, ensure_tensor
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

from .api_utils import (
    _dtype_to_str,
    _emit_creation_node,
    _ItemProtocol,
    _NameHolderProtocol,
    _ValueHolderProtocol,
)


@typing.runtime_checkable
class _ArrayBackendProtocol(typing.Protocol):
    """Protocol for backends exposing `.array(...)`."""

    def array(self, obj: OpValue, dtype: OpValue = None) -> OpValue:
        """Create backend array.

        Args:
            obj (OpValue): Input object.
            dtype (OpValue): Optional data type.

        Returns:
            OpValue: Created backend array.
        """
        ...


@typing.runtime_checkable
class _ConvertToNumpyBackendProtocol(typing.Protocol):
    """Protocol for backends exposing `.convert_to_numpy(...)`."""

    def convert_to_numpy(self, x: OpValue) -> OpValue:
        """Convert tensor to numpy array.

        Args:
            x (OpValue): Input tensor.

        Returns:
            OpValue: Converted numpy array.
        """
        ...


@typing.runtime_checkable
class _AsarrayBackendProtocol(typing.Protocol):
    """Protocol for backends exposing `.asarray(...)`."""

    def asarray(self, obj: OpValue) -> OpValue:
        """Convert object to backend array.

        Args:
            obj (OpValue): Input object.

        Returns:
            OpValue: Converted backend array.
        """
        ...


@typing.runtime_checkable
class _NumpyMethodProtocol(typing.Protocol):
    """Protocol for tensors exposing `.numpy()`."""

    def numpy(self) -> OpValue:
        """Convert tensor to numpy array.

        Returns:
            OpValue: Converted numpy array.
        """
        ...


def _is_array_backend(val: OpValue) -> typing.TypeGuard[_ArrayBackendProtocol]:
    """Check whether `val` exposes an `array` method.

    Args:
        val (OpValue): Active backend.

    Returns:
        bool: True if `val` satisfies `_ArrayBackendProtocol`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _ArrayBackendProtocol)
    return found or is_mock_with_attrs(val, "array")


def _is_convert_to_numpy_backend(val: OpValue) -> typing.TypeGuard[_ConvertToNumpyBackendProtocol]:
    """Check whether `val` exposes a `convert_to_numpy` method.

    Args:
        val (OpValue): Active backend.

    Returns:
        bool: True if `val` satisfies `_ConvertToNumpyBackendProtocol`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _ConvertToNumpyBackendProtocol)
    return found or is_mock_with_attrs(val, "convert_to_numpy")


def _is_asarray_backend(val: OpValue) -> typing.TypeGuard[_AsarrayBackendProtocol]:
    """Check whether `val` exposes an `asarray` method.

    Args:
        val (OpValue): Active backend.

    Returns:
        bool: True if `val` satisfies `_AsarrayBackendProtocol`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _AsarrayBackendProtocol)
    return found or is_mock_with_attrs(val, "asarray")


def _is_numpy_method_obj(val: OpValue) -> typing.TypeGuard[_NumpyMethodProtocol]:
    """Check whether `val` exposes a `numpy` method.

    Args:
        val (OpValue): Candidate tensor.

    Returns:
        bool: True if `val` satisfies `_NumpyMethodProtocol`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _NumpyMethodProtocol)
    return found or is_mock_with_attrs(val, "numpy")


def _can_read_data_holder(val: DataHolderProtocol) -> bool:
    """Verify accessing `val.data` does not raise `AttributeError`.

    Args:
        val (DataHolderProtocol): Candidate data holder.

    Returns:
        bool: `True` when `val.data` is readable.
    """
    try:
        _ = val.data
    except AttributeError:
        return False
    return True


def _is_data_holder_obj(val: OpValue) -> typing.TypeGuard[DataHolderProtocol]:
    """Check whether `val` exposes a `data` attribute.

    Args:
        val (OpValue): Candidate tensor wrapper.

    Returns:
        bool: True if `val` satisfies `DataHolderProtocol`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, DataHolderProtocol) and _can_read_data_holder(val)
    return found or is_mock_with_attrs(val, "data")


def _is_item_holder_obj(val: OpValue) -> typing.TypeGuard[_ItemProtocol]:
    """Check whether `val` exposes an `item` method.

    Args:
        val (OpValue): Candidate scalar array.

    Returns:
        bool: True if `val` satisfies `_ItemProtocol`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _ItemProtocol)
    return found or is_mock_with_attrs(val, "item")


def _is_value_holder_obj(val: OpValue) -> typing.TypeGuard[_ValueHolderProtocol]:
    """Check whether `val` exposes a `value` attribute.

    Args:
        val (OpValue): Candidate dtype object.

    Returns:
        bool: True if `val` satisfies `_ValueHolderProtocol`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _ValueHolderProtocol)
    return found or is_mock_with_attrs(val, "value")


def _unpack_shape(shape: Sequence[int | Tensor]) -> tuple[int | OpValue, ...]:
    """Compute the unpack shape tensor operation.

    Args:
        shape (Sequence[int | Tensor]): Input `shape` argument (shape).

    Returns:
        tuple[int | OpValue, ...]: Computed unpacked shape tuple.
    """
    unpacked_shape: list[int | OpValue] = []
    for s in shape:
        raw_s: OpValue = s
        if _is_data_holder_obj(raw_s):
            s_val: OpValue = raw_s.data
            if _is_item_holder_obj(s_val):
                s_val = s_val.item()
            try:
                unpacked_shape.append(coerce_int(s_val))
            except (TypeError, ValueError):
                unpacked_shape.append(s_val)
        elif _is_item_holder_obj(raw_s):
            unpacked_shape.append(coerce_int(raw_s.item()))
        else:
            try:
                unpacked_shape.append(coerce_int(raw_s))
            except (TypeError, ValueError):
                unpacked_shape.append(raw_s)
    return tuple(unpacked_shape)


def _infer_dtype(val_arr: OpValue) -> DType:
    """Infers the DType from a backend array.

    Args:
        val_arr (OpValue): Input `val_arr` argument (val arr).

    Returns:
        DType: Computed `DType` output.
    """
    dtype_str = str(extract_dtype(val_arr, "float32"))
    if dtype_str.startswith("<U") or dtype_str.startswith("|S"):
        return DType.String
    if dtype_str in ("Any", "object", "O"):
        return DType.Object
    return DType(dtype_str)


def _get_dtype_val(dtype: DType | str | type | None) -> str | type | None:
    """Get the backend dtype value.

    Args:
        dtype (DType | str | type | None): Input `dtype` argument (dtype).

    Returns:
        str | type | None: Computed `str | type | None` output.
    """
    if _is_value_holder_obj(dtype):
        return str(dtype.value)
    raw_dtype: OpValue | _NameHolderProtocol = dtype
    if isinstance(raw_dtype, _NameHolderProtocol):
        return str(raw_dtype)
    return dtype


def _try_create_array(
    backend: OpValue,
    obj: OpValue | Tensor | Sequence[OpValue],
    dtype_val: str | type | None = None,
) -> OpValue:
    """Try create array.

    Args:
        backend (OpValue): Input `backend` argument (backend).
        obj (OpValue | Tensor | Sequence[OpValue]): Input `obj` argument (obj).
        dtype_val (str | type | None): Input `dtype_val` argument (dtype val).

    Returns:
        OpValue: Computed `OpValue` output.
    """
    assert _is_array_backend(backend)
    if dtype_val is None:
        try:
            return backend.array(obj)
        except ValueError:
            return backend.array(obj, dtype="Any")
    try:
        return backend.array(obj, dtype=dtype_val)
    except TypeError:
        return backend.array(obj)


def _create_backend_array(
    obj: OpValue | Tensor | Sequence[OpValue],
    dtype: DType | str | type | None,
) -> OpValue:
    """Create the backend array.

    Args:
        obj (OpValue | Tensor | Sequence[OpValue]): Input `obj` argument (obj).
        dtype (DType | str | type | None): Input `dtype` argument (dtype).

    Returns:
        OpValue: Computed `OpValue` output.
    """
    backend = get_active_backend()
    dtype_val = None
    if dtype is not None:
        dtype_val = _get_dtype_val(dtype)

    res = _try_create_array(backend, obj, dtype_val)

    return res


def array(
    obj: OpValue,
    dtype: DType | None = None,
) -> Tensor:
    """Create an array.

    Args:
        obj (OpValue): Input `obj` argument (obj).
        dtype (DType | None): Input `dtype` argument (dtype).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    val_arr = _create_backend_array(obj, dtype)
    if dtype is None:
        dtype = _infer_dtype(val_arr)

    shape = extract_shape(val_arr)

    if config.eager_mode:
        return Tensor(val_arr, TensorConfig(shape, dtype, config.default_device))

    out_id = TracingNodeBuilder.extract_from_constant(val_arr)[0]

    return Tensor(
        ProxyTensor(coerce_str(out_id), shape, dtype.value), TensorConfig(shape, dtype, config.default_device)
    )


def asarray(
    a: OpValue,
    dtype: DType | None = None,
) -> Tensor:
    """Convert the input to an array.

    Args:
        a (OpValue): Input `a` argument (a).
        dtype (DType | None): Input `dtype` argument (dtype).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if isinstance(a, Tensor):
        if dtype is not None and a.dtype != dtype:
            from ml_switcheroo_compiler.ops.unary import cast as unary_cast

            return unary_cast(a, dtype)
        return a
    return array(a, dtype=dtype)


def convert_to_tensor(
    x: OpValue,
    dtype: DType | None = None,
) -> Tensor:
    """Convert the given object to a Tensor.

    Args:
        x (OpValue): Object to convert.
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
    unpacked_shape = _unpack_shape((shape,) if isinstance(shape, int) else tuple(shape))
    concrete_shape = tuple(coerce_int(d) for d in unpacked_shape if isinstance(d, int))

    if config.eager_mode:
        data = get_active_backend().execute_op(
            "Zeros",
            concrete_shape,
            dtype=_dtype_to_str(dtype),
        )
        return Tensor(data, TensorConfig(concrete_shape, dtype, device))
    return ensure_tensor(_emit_creation_node("Zeros", unpacked_shape, dtype, {}))


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
    unpacked_shape = _unpack_shape((shape,) if isinstance(shape, int) else tuple(shape))
    concrete_shape = tuple(coerce_int(d) for d in unpacked_shape if isinstance(d, int))

    if config.eager_mode:
        data = get_active_backend().execute_op(
            "Ones",
            concrete_shape,
            dtype=_dtype_to_str(dtype),
        )
        return Tensor(data, TensorConfig(concrete_shape, dtype, device))
    return ensure_tensor(_emit_creation_node("Ones", unpacked_shape, dtype, {}))


def _extract_fill_value(fill_value: float | int | bool | Tensor) -> float | int | bool:
    """Extract fill value.

    Args:
        fill_value (float | int | bool | Tensor): Input `fill_value` argument (fill value).

    Returns:
        float | int | bool: Computed `float | int | bool` output.
    """
    raw_fill: OpValue = fill_value
    val: OpValue = raw_fill.data if _is_data_holder_obj(raw_fill) else raw_fill
    if _is_item_holder_obj(val):
        val = val.item()
    assert isinstance(val, (float, int, bool))
    return val


def _full_eager(shape: tuple[int, ...], fill_value: OpValue, dtype: DType, device: Device) -> Tensor:
    """Full eager.

    Args:
        shape (tuple[int, ...]): Input `shape` argument (shape).
        fill_value (OpValue): Input `fill_value` argument (fill value).
        dtype (DType): Input `dtype` argument (dtype).
        device (Device): Input `device` argument (device).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    dt_val = _dtype_to_str(dtype)
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
        shape (int | Sequence[int]): Input `shape` argument (shape).
        fill_value (float): Input `fill_value` argument (fill value).
        dtype (DType | None): Input `dtype` argument (dtype).
        device (Device | None): Input `device` argument (device).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    unpacked_shape = _unpack_shape((shape,) if isinstance(shape, int) else tuple(shape))
    concrete_shape = tuple(coerce_int(d) for d in unpacked_shape if isinstance(d, int))
    fill_value = _extract_fill_value(fill_value)

    if config.eager_mode:
        return _full_eager(concrete_shape, fill_value, dtype, device)

    return ensure_tensor(
        _emit_creation_node(
            "Full",
            unpacked_shape,
            dtype,
            {"fill_value": fill_value},
        )
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
            dtype=_dtype_to_str(dtype),
        )
        return Tensor(data, TensorConfig(input.shape, dtype, device))
    return ensure_tensor(_emit_creation_node("ConstantOfShape", input.shape, dtype, {"value": 0}))


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
            dtype=_dtype_to_str(dtype),
        )
        return Tensor(data, TensorConfig(input.shape, dtype, device))
    return ensure_tensor(_emit_creation_node("ConstantOfShape", input.shape, dtype, {"value": 1}))


def full_like(
    input: Tensor,
    fill_value: float,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a tensor filled with `fill_value`, with the same size as `input`.

    Args:
        input (Tensor): Input `input` argument (input).
        fill_value (float): Input `fill_value` argument (fill value).
        dtype (DType | None): Input `dtype` argument (dtype).
        device (Device | None): Input `device` argument (device).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    dtype = dtype or input.dtype
    device = device or input.device
    if config.eager_mode:
        data = get_active_backend().execute_op(
            "Full_like",
            input.data,
            fill_value,
            dtype=_dtype_to_str(dtype),
        )
        return Tensor(data, TensorConfig(input.shape, dtype, device))
    return ensure_tensor(
        _emit_creation_node(
            "ConstantOfShape",
            input.shape,
            dtype,
            {"value": fill_value},
        )
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
    unpacked_shape = _unpack_shape((shape,) if isinstance(shape, int) else tuple(shape))
    concrete_shape = tuple(coerce_int(d) for d in unpacked_shape if isinstance(d, int))

    if config.eager_mode:
        data = get_active_backend().execute_op(
            "Empty",
            concrete_shape,
            dtype=_dtype_to_str(dtype),
        )
        return Tensor(data, TensorConfig(concrete_shape, dtype, device))
    return ensure_tensor(_emit_creation_node("Zeros", unpacked_shape, dtype, {}))


def empty_like(x: Tensor, dtype: DType | None = None) -> Tensor:
    """Return a new array with the same shape and type as a given array.

    Args:
        x (Tensor): The input x tensor.
        dtype (DType | None): The target data type.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    return empty(extract_shape(x), dtype=dtype if dtype is not None else x.dtype)


def convert_to_numpy(x: Tensor) -> OpValue:
    """Convert a tensor to a numpy array.

    Args:
        x (Tensor): Input tensor.

    Returns:
        OpValue: A numpy array.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend: OpValue = get_active_backend()
    if _is_convert_to_numpy_backend(backend):
        return backend.convert_to_numpy(x)
    raw_x: OpValue = x
    if _is_numpy_method_obj(raw_x):
        return raw_x.numpy()
    assert _is_asarray_backend(backend)
    return backend.asarray(extract_data(x))


def frombuffer(
    buffer: OpValue,
    dtype: DType | None = None,
    count: int = -1,
    offset: int = 0,
) -> Tensor:
    """Create a 1-D tensor from a buffer.

    Args:
        buffer (OpValue): An object that exposes the buffer interface.
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
            dtype=_dtype_to_str(dtype),
            count=count,
            offset=offset,
        )
        shape = extract_shape(data)
        return Tensor(data, TensorConfig(shape, dtype, config.default_device))
    return ensure_tensor(_emit_creation_node("Frombuffer", (count,) if count != -1 else (), dtype, {"offset": offset}))
