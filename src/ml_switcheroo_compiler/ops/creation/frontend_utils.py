"""Module frontend_utils.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Constants & Creation Operations."""
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
    attributes: dict[str, object] | None = None,
) -> object:
    """Emit a creation node to the IR graph.

    Args:
        op_type (str): The op_type parameter.
        shape (Sequence): The shape parameter.
        dtype (DType): The dtype parameter.
        attributes (object): The attributes parameter.

    Returns:
        Tensor: Result.

    Raises:
        RuntimeError: An exception.
    """
    if not global_tracing_state.is_tracing:
        msg: object = f"Cannot emit {op_type} node outside of a tracing context."
        raise RuntimeError(msg)

    out_id: object = str(uuid.uuid4())
    node: object = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=[],
        attributes={**(attributes or {}), "dtype": dtype.value},
        shape_metadata=shape,
    )
    global_tracing_state.add_node(node)

    proxy: object = ProxyTensor(
        id=out_id,
        shape=shape,
        dtype=dtype.value if hasattr(dtype, "value") else getattr(dtype, "name", str(dtype)),
    )
    return Tensor(proxy, TensorConfig(shape, dtype, config.default_device))


def _emit_constant_node(
    value: object,
    dtype: DType,
) -> object:
    """Emit a Constant node to the IR graph.

    Args:
        value (object): The value parameter.
        dtype (DType): The dtype parameter.

    Returns:
        Tensor: Result.

    Raises:
        RuntimeError: An exception.
    """
    if not global_tracing_state.is_tracing:
        msg: object = "Cannot emit Constant node outside of a tracing context."
        raise RuntimeError(msg)

    out_id: object = str(uuid.uuid4())
    val_arr: object = get_active_backend().array(value, dtype=dtype.value if hasattr(dtype, "value") else getattr(dtype, "name", str(dtype)))
    shape: object = tuple(val_arr.shape)

    node: object = LogicalNode(
        id=out_id,
        op_type="Constant",
        inputs=[],
        attributes={"value": val_arr.tolist() if val_arr.ndim > 0 else val_arr.item()},
        shape_metadata=shape,
    )
    global_tracing_state.add_node(node)

    proxy: object = ProxyTensor(
        id=out_id,
        shape=shape,
        dtype=dtype.value if hasattr(dtype, "value") else getattr(dtype, "name", str(dtype)),
    )
    return Tensor(proxy, TensorConfig(shape, dtype, config.default_device))


@register_op("FromDlpack")
class FromDlpack(OpDef):
    """Create a switcheroo array from a DLPack capsule."""

    op_name: object = "FromDlpack"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        obj: object = args[0] if len(args) > 0 else None
        return getattr(obj, "shape", ())


@register_op("Frompyfunc")
class Frompyfunc(OpDef):
    """Take an arbitrary Python function and returns a NumPy ufunc."""

    op_name: object = "Frompyfunc"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("Geomspace")
class Geomspace(OpDef):
    """Return numbers spaced evenly on a log scale (a geometric progression)."""

    op_name: object = "Geomspace"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        start: object = args[0] if len(args) > 0 else None
        stop: object = args[1] if len(args) > 1 else None
        num: object = kwargs.get("num", args[2] if len(args) > 2 else 50)
        axis: object = kwargs.get("axis", 0)
        shape1: object = getattr(start, "shape", ())
        shape2: object = getattr(stop, "shape", ())
        b_shape: object = shape1 if len(shape1) > len(shape2) else shape2
        if not b_shape:
            return (num,)

        out_shape: object = list(b_shape)
        insert_axis: object = axis + len(b_shape) + 1 if axis < 0 else axis
        out_shape.insert(insert_axis, num)
        return tuple(out_shape)


@register_op("Geometric")
class Geometric(OpDef):
    """Draw samples from the geometric distribution."""

    op_name: object = "Geometric"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        p: object = args[0] if len(args) > 0 else None
        size: object = kwargs.get("size", args[1] if len(args) > 1 else None)
        if size is None:
            return getattr(p, "shape", ())
        if isinstance(size, int):
            return (size,)
        return tuple(size)


def from_dlpack(obj: object) -> object:
    """Create a switcheroo array from a DLPack capsule.

    Args:
        obj (object): The obj parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("FromDlpack", obj)


def frompyfunc(func: object, nin: int, nout: int) -> object:
    """Take an arbitrary Python function and returns a NumPy ufunc.

    Args:
        func (object): The func parameter.
        nin (int): The nin parameter.
        nout (int): The nout parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Frompyfunc", func, nin, nout)


def geomspace(start: object, stop: object, num: int = 50, endpoint: bool = True, dtype: object = None, axis: int = 0) -> object:
    """Return numbers spaced evenly on a log scale (a geometric progression).

    Args:
        start (object): The start parameter.
        stop (object): The stop parameter.
        num (int): The num parameter.
        endpoint (bool): The endpoint parameter.
        dtype (object): The dtype parameter.
        axis (int): The axis parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Geomspace", start, stop, num=num, endpoint=endpoint, dtype=dtype, axis=axis)


def geometric(p: object, size: object = None) -> object:
    """Draw samples from the geometric distribution.

    Args:
        p (object): The p parameter.
        size (object): The size parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Geometric", p, size=size)
