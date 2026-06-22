"""Constants & Creation Operations."""

from __future__ import annotations


import uuid
from typing import TYPE_CHECKING

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing import ProxyTensor, _tracer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType


def _emit_creation_node(
    op_type: str,
    shape: Sequence[int],
    dtype: DType,
    attributes: dict | None = None,
) -> Tensor:
    """Emit a creation node to the IR graph.

    Args:
        op_type (str): The op_type parameter for the operation.
        shape (Sequence[int]): The target shape.
        dtype (DType): The target data type.
        attributes (dict | None): The attributes parameter for the operation.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    if not _tracer.is_tracing:
        msg = f"Cannot emit {op_type} node outside of a tracing context."
        raise RuntimeError(msg)

    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=[],
        attributes={**(attributes or {}), "dtype": dtype.value},
        shape_metadata=shape,
    )
    _tracer.add_node(node)

    proxy = ProxyTensor(id=out_id, shape=shape, dtype=dtype.value)
    return Tensor(proxy, TensorConfig(shape, dtype, config.default_device))


def _emit_constant_node(
    value: object,
    dtype: DType,
) -> Tensor:
    """Emit a Constant node to the IR graph.

    Args:
        value (object): The value parameter for the operation.
        dtype (DType): The target data type.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    if not _tracer.is_tracing:
        msg = "Cannot emit Constant node outside of a tracing context."
        raise RuntimeError(msg)

    out_id = str(uuid.uuid4())
    val_arr = get_active_backend().array(value, dtype=dtype.value)
    shape = tuple(val_arr.shape)

    node = LogicalNode(
        id=out_id,
        op_type="Constant",
        inputs=[],
        attributes={"value": val_arr.tolist() if val_arr.ndim > 0 else val_arr.item()},
        shape_metadata=shape,
    )
    _tracer.add_node(node)

    proxy = ProxyTensor(id=out_id, shape=shape, dtype=dtype.value)
    return Tensor(proxy, TensorConfig(shape, dtype, config.default_device))


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


def arange(
    start: float = 0,
    stop: float | None = None,
    step: float = 1,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a 1-D tensor of size with values from the interval `[start, stop)`.

    Args:
        start (Union[float, int]): Argument start
        stop (Optional[Union[float, int]]): Argument stop
        step (Union[float, int]): Argument step
        dtype (Optional[DType]): The data type
        device (Optional[Device]): The device to store the tensor on.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    if stop is None:
        stop = start
        start = 0
    import math

    size = max(math.ceil((stop - start) / step), 0)
    shape = (size,)

    if config.eager_mode:
        data = get_active_backend().execute_op("Arange", start, stop, step, dtype=dtype.value)
        return Tensor(data, TensorConfig(shape, dtype, device))
    return _emit_creation_node(
        "Arange",
        shape,
        dtype,
        {"start": start, "stop": stop, "step": step},
    )


def linspace(
    start: float,
    stop: float,
    steps: int,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Create a 1D tensor evenly spaced from `start` to `stop`.

    Args:
        start (Union[float, int]): Argument start
        stop (Union[float, int]): Argument stop
        steps (int): Argument steps
        dtype (Optional[DType]): The data type
        device (Optional[Device]): The device to store the tensor on.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    shape = (steps,)

    if config.eager_mode:
        data = get_active_backend().execute_op("Linspace", start, stop, steps, dtype=dtype.value)
        return Tensor(data, TensorConfig(shape, dtype, device))
    return _emit_creation_node(
        "LinSpace",
        shape,
        dtype,
        {"start": start, "stop": stop, "steps": steps},
    )


def eye(
    n: int,
    m: int | None = None,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a 2-D tensor with ones on the diagonal and zeros elsewhere.

    Args:
        n (int): Argument n
        m (Optional[int]): Argument m
        dtype (Optional[DType]): The data type
        device (Optional[Device]): The device to store the tensor on.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    m = m if m is not None else n
    shape = (n, m)

    if config.eager_mode:
        data = get_active_backend().execute_op("Eye", n, m, dtype=dtype.value)
        return Tensor(data, TensorConfig(shape, dtype, device))
    return _emit_creation_node("EyeLike", shape, dtype, {"n": n, "m": m})


def identity(
    n: int,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return the 2-D identity matrix of shape `(n, n)`.

    Args:
        n (int): Argument n
        dtype (Optional[DType]): The data type
        device (Optional[Device]): The device to store the tensor on.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    return eye(n, n, dtype, device)


def diag(input: Tensor, diagonal: int = 0) -> Tensor:
    """Return a 2-D square tensor with diagonal, or extracts diagonal.

    Args:
        input (Tensor): The input tensor
        diagonal (int): Argument diagonal

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    device = getattr(input, "device", None)
    dtype = getattr(input, "dtype", None)

    if config.eager_mode:
        data = get_active_backend().execute_op("Diag", getattr(input, "data", input), k=diagonal)
        shape = data.shape if hasattr(data, "shape") else ()
        from ml_switcheroo_compiler.core.dtype import DType

        if dtype is None:
            dtype = getattr(data, "dtype", DType.Float32)
        return Tensor(data, TensorConfig(shape, dtype, device))
    if len(input.shape) == 1:
        n = input.shape[0] + abs(diagonal)
        shape = (n, n)
    elif len(input.shape) == 2:
        n = min(input.shape) - abs(diagonal)
        shape = (max(0, n),)
    else:
        msg = "diag requires a 1D or 2D tensor."
        raise ValueError(msg)

    if not _tracer.is_tracing:
        msg = "Cannot emit diag node outside of a tracing context."
        raise RuntimeError(msg)
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Diag",
        inputs=[input.data.id],
        attributes={"k": diagonal},
        shape_metadata=shape,
    )
    _tracer.add_node(node)
    proxy = ProxyTensor(id=out_id, shape=shape, dtype=dtype.value)
    return Tensor(proxy, TensorConfig(shape, dtype, device))


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


def rand(
    *size: int,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a tensor filled with random numbers from a uniform distribution.

    Args:
        *size: Additional arguments.
        dtype (DType | None): The target data type.
        device (Device | None): The device parameter for the operation.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    shape = tuple(size)

    if config.eager_mode:
        data = get_active_backend().execute_op("Rand", *shape)
        return Tensor(data, TensorConfig(shape, dtype, device))
    return _emit_creation_node("Rand", shape, dtype)


def randn(
    *size: int,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a tensor filled with random numbers from a standard normal distribution.

    Args:
        *size: Additional arguments.
        dtype (DType | None): The target data type.
        device (Device | None): The device parameter for the operation.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    shape = tuple(size)

    if config.eager_mode:
        data = get_active_backend().execute_op("Randn", *shape)
        return Tensor(data, TensorConfig(shape, dtype, device))
    return _emit_creation_node("Randn", shape, dtype)


def randint(
    low: int,
    high: int,
    size: Sequence[int],
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a tensor filled with random integers from [low, high).

    Args:
        low (int): The low parameter for the operation.
        high (int): The high parameter for the operation.
        size (Sequence[int]): The size parameter for the operation.
        dtype (DType | None): The target data type.
        device (Device | None): The device parameter for the operation.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    dtype = dtype or config.default_int_dtype
    device = device or config.default_device
    shape = tuple(size)

    if config.eager_mode:
        data = get_active_backend().execute_op("Randint", low, high, size=shape)
        return Tensor(data, TensorConfig(shape, dtype, device))
    return _emit_creation_node("Randint", shape, dtype, {"low": low, "high": high})


def manual_seed(seed: int) -> int:
    """Sets the seed for generating random numbers.

    Args:
        seed (int): The random seed.

    Returns:
        int: The evaluated output resulting from this operation.
    """
    if config.eager_mode:
        get_active_backend().execute_op("Seed", seed)
        return seed
    _emit_creation_node("ManualSeed", (), config.default_int_dtype, {"seed": seed})
    return seed


def blackman(M: int) -> Tensor:
    """Return the blackman window.

    Args:
        M (int): Number of points in the output window.

    Returns:
        Tensor: The window.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        data = get_active_backend().execute_op("Blackman", M)
        from ml_switcheroo_compiler.core.dtype import DType

        return Tensor(data, TensorConfig((M,), DType.Float32, None))
    return _emit_creation_node("Blackman", (M,), DType.Float32, {})


def bartlett(M: int) -> Tensor:
    """Return the bartlett window.

    Args:
        M (int): Number of points in the output window.

    Returns:
        Tensor: The window.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        data = get_active_backend().execute_op("Bartlett", M)
        from ml_switcheroo_compiler.core.dtype import DType

        return Tensor(data, TensorConfig((M,), DType.Float32, None))
    return _emit_creation_node("Bartlett", (M,), DType.Float32, {})


def hamming(M: int) -> Tensor:
    """Return the hamming window.

    Args:
        M (int): Number of points in the output window.

    Returns:
        Tensor: The window.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        data = get_active_backend().execute_op("Hamming", M)
        from ml_switcheroo_compiler.core.dtype import DType

        return Tensor(data, TensorConfig((M,), DType.Float32, None))
    return _emit_creation_node("Hamming", (M,), DType.Float32, {})


def hanning(M: int) -> Tensor:
    """Return the hanning window.

    Args:
        M (int): Number of points in the output window.

    Returns:
        Tensor: The window.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        data = get_active_backend().execute_op("Hanning", M)
        from ml_switcheroo_compiler.core.dtype import DType

        return Tensor(data, TensorConfig((M,), DType.Float32, None))
    return _emit_creation_node("Hanning", (M,), DType.Float32, {})


def kaiser(M: int, beta: float) -> Tensor:
    """Return the Kaiser window.

    Args:
        M (int): Number of points in the output window.
        beta (float): Shape parameter.

    Returns:
        Tensor: The window.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        data = get_active_backend().execute_op("Kaiser", M, beta)
        from ml_switcheroo_compiler.core.dtype import DType

        return Tensor(data, TensorConfig((M,), DType.Float32, None))
    return _emit_creation_node("Kaiser", (M,), DType.Float32, {"beta": beta})
