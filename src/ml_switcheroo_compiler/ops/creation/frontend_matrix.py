"""Constants & Creation Operations."""

from __future__ import annotations

import uuid

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.constants import MATRIX_RANK
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar, _get_attr, _has_attr
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state

from .frontend_utils import _emit_creation_node


def eye(
    n: int,
    m: int | None = None,
    k: int = 0,
    dtype: DType | None = None,
    device: Device | None = None,
) -> TensorOrScalar:
    """Return a 2-D tensor with ones on the diagonal and zeros elsewhere.

    Args:
        n (int): The n parameter.
        m (Optional[int]): The m parameter.
        k (int): Index of the diagonal.
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
        data = get_active_backend().execute_op(
            "Eye",
            n,
            m,
            k=k,
            dtype=dtype.value if _has_attr(dtype, "value") else _get_attr(dtype, "name", str(dtype)),
        )
        return Tensor(data, TensorConfig(shape, dtype, device))
    return _emit_creation_node("EyeLike", shape, dtype, {"n": n, "m": m, "k": k})


def identity(
    n: int,
    dtype: DType | None = None,
    device: Device | None = None,
) -> TensorOrScalar:
    """Return the 2-D identity matrix of shape `(n, n)`.

    Args:
        n (int): The n parameter.
        dtype (Optional[DType]): The data type
        device (Optional[Device]): The device to store the tensor on.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    return eye(n, n, 0, dtype, device)


def _diag_eager(input: Tensor, diagonal: int, device: TensorOrScalar, dtype: TensorOrScalar) -> TensorOrScalar:
    """Compute the diag eager tensor operation.

    Args:
        input (Tensor): The input parameter.
        diagonal (int): The diagonal parameter.
        device (TensorOrScalar): The device parameter.
        dtype (TensorOrScalar): The dtype parameter.

    Returns:
        Tensor: Result.
    """
    data = get_active_backend().execute_op("Diag", _get_attr(input, "data", input), k=diagonal)
    shape = _get_attr(data, "shape") if _has_attr(data, "shape") else ()
    if dtype is None:
        dtype = _get_attr(data, "dtype", DType.Float32)
    return Tensor(data, TensorConfig(shape, dtype, device))


def diag(input: Tensor, diagonal: int = 0) -> TensorOrScalar:
    """Return a 2-D square tensor with diagonal, or extracts diagonal.

    Args:
        input (Tensor): The input parameter.
        diagonal (int): The diagonal parameter.

    Returns:
        Tensor: Result.

    Raises:
        RuntimeError: An exception.
        ValueError: An exception.
    """
    device = _get_attr(input, "device", None)
    dtype = _get_attr(input, "dtype", None)

    if config.eager_mode:
        return _diag_eager(input, diagonal, device, dtype)

    input_shape = _get_attr(input, "shape", ())
    if len(input_shape) == MATRIX_RANK:
        n = min(input_shape) - abs(diagonal)
        shape = (max(0, n),)
    else:
        msg = "diag requires a 1D or 2D tensor."
        raise ValueError(msg)

    if not global_tracing_state.is_tracing:
        msg = "Cannot emit diag node outside of a tracing context."
        raise RuntimeError(msg)
    out_id = str(uuid.uuid4())
    input_id = _get_attr(input.data, "id") if _has_attr(input, "data") else _get_attr(input, "id", "const")
    node = LogicalNode(
        id=out_id,
        op_type="Diag",
        inputs=[input_id],
        attributes={"k": diagonal},
        shape_metadata=shape,
    )
    global_tracing_state.add_node(node)
    proxy = ProxyTensor(
        id=out_id,
        shape=shape,
        dtype=_get_attr(dtype, "value") if _has_attr(dtype, "value") else _get_attr(dtype, "name", str(dtype)),
    )
    return Tensor(proxy, TensorConfig(shape, dtype, device))
