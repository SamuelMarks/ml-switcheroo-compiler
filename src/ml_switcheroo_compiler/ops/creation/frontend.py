"""Constants & Creation Operations."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import numpy as np
from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor
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
    """Emit a creation node to the IR graph."""
    if not _tracer.is_tracing:
        msg = f"Cannot emit {op_type} node outside of a tracing context."
        raise RuntimeError(msg)

    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=[],
        attributes=attributes or {},
        shape_metadata=shape,
    )
    _tracer.add_node(node)

    proxy = ProxyTensor(id=out_id, shape=shape, dtype=dtype.value)
    return Tensor(data=proxy, shape=shape, dtype=dtype, device=config.default_device)


def zeros(
    shape: int | Sequence[int],
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a tensor filled with the scalar value 0.

    Args:
    shape (Union[int, Sequence[int]]): Argument shape
    dtype (Optional[DType]): The data type
    device (Optional[Device]): Argument device
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    shape = (shape,) if isinstance(shape, int) else tuple(shape)

    if config.eager_mode:
        data = np.zeros(shape, dtype=dtype.value)
        return Tensor(data, shape, dtype, device)
    return _emit_creation_node("ConstantOfShape", shape, dtype, {"value": 0})


def ones(
    shape: int | Sequence[int],
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a tensor filled with the scalar value 1.

    Args:
    shape (Union[int, Sequence[int]]): Argument shape
    dtype (Optional[DType]): The data type
    device (Optional[Device]): Argument device
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    shape = (shape,) if isinstance(shape, int) else tuple(shape)

    if config.eager_mode:
        data = np.ones(shape, dtype=dtype.value)
        return Tensor(data, shape, dtype, device)
    return _emit_creation_node("ConstantOfShape", shape, dtype, {"value": 1})


def full(
    shape: int | Sequence[int],
    fill_value: float,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a tensor filled with `fill_value`.

    Args:
    shape (Union[int, Sequence[int]]): Argument shape
    fill_value (Union[float, int]): Argument fill_value
    dtype (Optional[DType]): The data type
    device (Optional[Device]): Argument device
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    shape = (shape,) if isinstance(shape, int) else tuple(shape)

    if config.eager_mode:
        data = np.full(shape, fill_value, dtype=dtype.value)
        return Tensor(data, shape, dtype, device)
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
    device (Optional[Device]): Argument device
    """
    dtype = dtype or input.dtype
    device = device or input.device
    if config.eager_mode:
        data = np.zeros_like(input.data, dtype=dtype.value)
        return Tensor(data, input.shape, dtype, device)
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
    device (Optional[Device]): Argument device
    """
    dtype = dtype or input.dtype
    device = device or input.device
    if config.eager_mode:
        data = np.ones_like(input.data, dtype=dtype.value)
        return Tensor(data, input.shape, dtype, device)
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
    fill_value (Union[float, int]): Argument fill_value
    dtype (Optional[DType]): The data type
    device (Optional[Device]): Argument device
    """
    dtype = dtype or input.dtype
    device = device or input.device
    if config.eager_mode:
        data = np.full_like(input.data, fill_value, dtype=dtype.value)
        return Tensor(data, input.shape, dtype, device)
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
    device (Optional[Device]): Argument device
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
        data = np.arange(start, stop, step, dtype=dtype.value)
        return Tensor(data, shape, dtype, device)
    return _emit_creation_node(
        "Range",
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
    device (Optional[Device]): Argument device
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    shape = (steps,)

    if config.eager_mode:
        data = np.linspace(start, stop, steps, dtype=dtype.value)
        return Tensor(data, shape, dtype, device)
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
    device (Optional[Device]): Argument device
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    m = m if m is not None else n
    shape = (n, m)

    if config.eager_mode:
        data = np.eye(n, m, dtype=dtype.value)
        return Tensor(data, shape, dtype, device)
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
    device (Optional[Device]): Argument device
    """
    return eye(n, n, dtype, device)


def diag(input: Tensor, diagonal: int = 0) -> Tensor:
    """Return a 2-D square tensor with diagonal, or extracts diagonal.

    Args:
    input (Tensor): The input tensor
    diagonal (int): Argument diagonal
    """
    device = input.device
    dtype = input.dtype

    if config.eager_mode:
        data = np.diag(input.data, k=diagonal)
        return Tensor(data, data.shape, dtype, device)
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
    return Tensor(data=proxy, shape=shape, dtype=dtype, device=device)


def empty(
    shape: int | Sequence[int],
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Return a tensor filled with uninitialized data.

    Args:
    shape (Union[int, Sequence[int]]): Argument shape
    dtype (Optional[DType]): The data type
    device (Optional[Device]): Argument device
    """
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    shape = (shape,) if isinstance(shape, int) else tuple(shape)

    if config.eager_mode:
        data = np.empty(shape, dtype=dtype.value)
        return Tensor(data, shape, dtype, device)
    return _emit_creation_node("ConstantOfShape", shape, dtype, {"value": 0})
