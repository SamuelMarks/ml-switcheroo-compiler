"""Constants & Creation Operations."""

from typing import Union, Sequence, Optional
import uuid
import numpy as np
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.device import Device
from ml_switcheroo.core.config import config
from ml_switcheroo.tracing import _tracer, ProxyTensor
from ml_switcheroo_ir import LogicalNode


def _emit_creation_node(
    op_type: str, shape: Sequence[int], dtype: DType, attributes: dict = None
) -> Tensor:
    """Emit a creation node to the IR graph."""
    if not _tracer.is_tracing:
        raise RuntimeError(f"Cannot emit {op_type} node outside of a tracing context.")

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
    shape: Union[int, Sequence[int]],
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    """Returns a tensor filled with the scalar value 0."""
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    shape = (shape,) if isinstance(shape, int) else tuple(shape)

    if config.eager_mode:
        data = np.zeros(shape, dtype=dtype.value)
        return Tensor(data, shape, dtype, device)
    else:
        return _emit_creation_node("ConstantOfShape", shape, dtype, {"value": 0})


def ones(
    shape: Union[int, Sequence[int]],
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    """Returns a tensor filled with the scalar value 1."""
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    shape = (shape,) if isinstance(shape, int) else tuple(shape)

    if config.eager_mode:
        data = np.ones(shape, dtype=dtype.value)
        return Tensor(data, shape, dtype, device)
    else:
        return _emit_creation_node("ConstantOfShape", shape, dtype, {"value": 1})


def full(
    shape: Union[int, Sequence[int]],
    fill_value: Union[float, int],
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    """Returns a tensor filled with `fill_value`."""
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    shape = (shape,) if isinstance(shape, int) else tuple(shape)

    if config.eager_mode:
        data = np.full(shape, fill_value, dtype=dtype.value)
        return Tensor(data, shape, dtype, device)
    else:
        return _emit_creation_node(
            "ConstantOfShape", shape, dtype, {"value": fill_value}
        )


def zeros_like(
    input: Tensor, dtype: Optional[DType] = None, device: Optional[Device] = None
) -> Tensor:
    """Returns a tensor filled with the scalar value 0, with the same size as `input`."""
    dtype = dtype or input.dtype
    device = device or input.device
    if config.eager_mode:
        data = np.zeros_like(input.data, dtype=dtype.value)
        return Tensor(data, input.shape, dtype, device)
    else:
        return _emit_creation_node("ConstantOfShape", input.shape, dtype, {"value": 0})


def ones_like(
    input: Tensor, dtype: Optional[DType] = None, device: Optional[Device] = None
) -> Tensor:
    """Returns a tensor filled with the scalar value 1, with the same size as `input`."""
    dtype = dtype or input.dtype
    device = device or input.device
    if config.eager_mode:
        data = np.ones_like(input.data, dtype=dtype.value)
        return Tensor(data, input.shape, dtype, device)
    else:
        return _emit_creation_node("ConstantOfShape", input.shape, dtype, {"value": 1})


def full_like(
    input: Tensor,
    fill_value: Union[float, int],
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    """Returns a tensor filled with `fill_value`, with the same size as `input`."""
    dtype = dtype or input.dtype
    device = device or input.device
    if config.eager_mode:
        data = np.full_like(input.data, fill_value, dtype=dtype.value)
        return Tensor(data, input.shape, dtype, device)
    else:
        return _emit_creation_node(
            "ConstantOfShape", input.shape, dtype, {"value": fill_value}
        )


def arange(
    start: Union[float, int] = 0,
    stop: Optional[Union[float, int]] = None,
    step: Union[float, int] = 1,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    """Returns a 1-D tensor of size with values from the interval `[start, stop)`."""
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
    else:
        return _emit_creation_node(
            "Range", shape, dtype, {"start": start, "stop": stop, "step": step}
        )


def linspace(
    start: Union[float, int],
    stop: Union[float, int],
    steps: int,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    """Creates a one-dimensional tensor of size `steps` whose values are evenly spaced from `start` to `stop`."""
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    shape = (steps,)

    if config.eager_mode:
        data = np.linspace(start, stop, steps, dtype=dtype.value)
        return Tensor(data, shape, dtype, device)
    else:
        return _emit_creation_node(
            "LinSpace", shape, dtype, {"start": start, "stop": stop, "steps": steps}
        )


def eye(
    n: int,
    m: Optional[int] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    """Returns a 2-D tensor with ones on the diagonal and zeros elsewhere."""
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    m = m if m is not None else n
    shape = (n, m)

    if config.eager_mode:
        data = np.eye(n, m, dtype=dtype.value)
        return Tensor(data, shape, dtype, device)
    else:
        return _emit_creation_node("EyeLike", shape, dtype, {"n": n, "m": m})


def identity(
    n: int, dtype: Optional[DType] = None, device: Optional[Device] = None
) -> Tensor:
    """Returns the 2-D identity matrix of shape `(n, n)`."""
    return eye(n, n, dtype, device)


def diag(input: Tensor, diagonal: int = 0) -> Tensor:
    """Returns a 2-D square tensor with elements of `input` as diagonal, or extracts diagonal from a 2-D tensor."""
    device = input.device
    dtype = input.dtype

    if config.eager_mode:
        data = np.diag(input.data, k=diagonal)
        return Tensor(data, data.shape, dtype, device)
    else:
        if len(input.shape) == 1:
            n = input.shape[0] + abs(diagonal)
            shape = (n, n)
        elif len(input.shape) == 2:
            n = min(input.shape) - abs(diagonal)
            shape = (max(0, n),)
        else:
            raise ValueError("diag requires a 1D or 2D tensor.")

        if not _tracer.is_tracing:
            raise RuntimeError("Cannot emit diag node outside of a tracing context.")
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
    shape: Union[int, Sequence[int]],
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    """Returns a tensor filled with uninitialized data."""
    dtype = dtype or config.default_float_dtype
    device = device or config.default_device
    shape = (shape,) if isinstance(shape, int) else tuple(shape)

    if config.eager_mode:
        data = np.empty(shape, dtype=dtype.value)
        return Tensor(data, shape, dtype, device)
    else:
        return _emit_creation_node("ConstantOfShape", shape, dtype, {"value": 0})
