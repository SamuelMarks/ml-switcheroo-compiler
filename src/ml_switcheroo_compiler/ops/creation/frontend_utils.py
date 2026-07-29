"""Constants & Creation Operations."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state


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
    if not global_tracing_state.is_tracing:
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
    global_tracing_state.add_node(node)

    proxy = ProxyTensor(
        id=out_id,
        shape=shape,
        dtype=dtype.value if hasattr(dtype, "value") else getattr(dtype, "name", str(dtype)),
    )
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
    if not global_tracing_state.is_tracing:
        msg = "Cannot emit Constant node outside of a tracing context."
        raise RuntimeError(msg)

    out_id = str(uuid.uuid4())
    val_arr = get_active_backend().array(value, dtype=dtype.value if hasattr(dtype, "value") else getattr(dtype, "name", str(dtype)))
    shape = tuple(val_arr.shape)

    node = LogicalNode(
        id=out_id,
        op_type="Constant",
        inputs=[],
        attributes={"value": val_arr.tolist() if val_arr.ndim > 0 else val_arr.item()},
        shape_metadata=shape,
    )
    global_tracing_state.add_node(node)

    proxy = ProxyTensor(
        id=out_id,
        shape=shape,
        dtype=dtype.value if hasattr(dtype, "value") else getattr(dtype, "name", str(dtype)),
    )
    return Tensor(proxy, TensorConfig(shape, dtype, config.default_device))


@register_op("FromDlpack")
class FromDlpack(OpDef):
    """Create a switcheroo array from a DLPack capsule."""

    op_name = "FromDlpack"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        obj = args[0] if len(args) > 0 else None
        return getattr(obj, "shape", ())


@register_op("Frompyfunc")
class Frompyfunc(OpDef):
    """Takes an arbitrary Python function and returns a NumPy ufunc."""

    op_name = "Frompyfunc"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Geomspace")
class Geomspace(OpDef):
    """Return numbers spaced evenly on a log scale (a geometric progression)."""

    op_name = "Geomspace"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        start = args[0] if len(args) > 0 else None
        stop = args[1] if len(args) > 1 else None
        num = kwargs.get("num", args[2] if len(args) > 2 else 50)
        axis = kwargs.get("axis", 0)
        shape1 = getattr(start, "shape", ())
        shape2 = getattr(stop, "shape", ())
        b_shape = shape1 if len(shape1) > len(shape2) else shape2
        if not b_shape:
            return (num,)

        out_shape = list(b_shape)
        insert_axis = axis + len(b_shape) + 1 if axis < 0 else axis
        out_shape.insert(insert_axis, num)
        return tuple(out_shape)


@register_op("Geometric")
class Geometric(OpDef):
    """Draw samples from the geometric distribution."""

    op_name = "Geometric"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        p = args[0] if len(args) > 0 else None
        size = kwargs.get("size", args[1] if len(args) > 1 else None)
        if size is None:
            return getattr(p, "shape", ())
        if isinstance(size, int):
            return (size,)
        return tuple(size)


def from_dlpack(obj: object) -> object:
    """Create a switcheroo array from a DLPack capsule."""
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("FromDlpack", obj)


def frompyfunc(func: object, nin: int, nout: int) -> object:
    """Takes an arbitrary Python function and returns a NumPy ufunc."""
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Frompyfunc", func, nin, nout)


def geomspace(start: object, stop: object, num: int = 50, endpoint: bool = True, dtype: object = None, axis: int = 0) -> object:
    """Return numbers spaced evenly on a log scale (a geometric progression)."""
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Geomspace", start, stop, num=num, endpoint=endpoint, dtype=dtype, axis=axis)


def geometric(p: object, size: object = None) -> object:
    """Draw samples from the geometric distribution."""
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Geometric", p, size=size)
