"""Creation node emission utilities and DLPack/ufunc/geometric creation operations."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, as_eager_list, as_int, get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import as_attrs
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state


def _emit_creation_node(
    op_type: str,
    shape: Sequence[int],
    dtype: DType,
    attributes: TensorOrScalar = None,
) -> TensorOrScalar:
    """Emit a symbolic tensor creation node to the active tracing IR graph.

    Args:
        op_type (str): Canonical IR creation operation type name.
        shape (Sequence[int]): Output tensor shape.
        dtype (DType): Output tensor data type.
        attributes (TensorOrScalar): Optional dictionary of node attributes.

    Returns:
        TensorOrScalar: Symbolic `Tensor` wrapping the emitted `LogicalNode`.

    Raises:
        RuntimeError: If called outside of an active tracing context.
    """
    if not global_tracing_state.is_tracing:
        msg = f"Cannot emit {op_type} node outside of a tracing context."
        raise RuntimeError(msg)

    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=[],
        attributes={**as_attrs(attributes or {}), "dtype": dtype.value},
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
    value: TensorOrScalar,
    dtype: DType,
) -> TensorOrScalar:
    """Emit a symbolic `Constant` node to the active tracing IR graph.

    Args:
        value (TensorOrScalar): Constant scalar or sequence value.
        dtype (DType): Output tensor data type.

    Returns:
        TensorOrScalar: Symbolic `Tensor` wrapping the emitted `Constant` node.

    Raises:
        RuntimeError: If called outside of an active tracing context.
    """
    if not global_tracing_state.is_tracing:
        msg = "Cannot emit Constant node outside of a tracing context."
        raise RuntimeError(msg)

    out_id = str(uuid.uuid4())
    val_arr = get_active_backend().array(
        value,
        dtype=dtype.value if hasattr(dtype, "value") else getattr(dtype, "name", str(dtype)),
    )
    shape = tuple(getattr(val_arr, "shape"))

    node = LogicalNode(
        id=out_id,
        op_type="Constant",
        inputs=[],
        attributes={"value": getattr(val_arr, "tolist")() if getattr(val_arr, "ndim") > 0 else getattr(val_arr, "item")()},
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
    """Create a compiler tensor from an external object implementing the DLPack protocol."""

    op_name = "FromDlpack"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer shape from the DLPack source tensor.

        Args:
            *args (EagerTensorProtocol): Positional DLPack source tensor arguments.
            **kwargs (TensorOrScalar): Keyword arguments (`ext_tensor` or `tensor`).

        Returns:
            tuple[int, ...]: Inferred tensor shape tuple.
        """
        obj = args[0] if len(args) > 0 else kwargs.get("ext_tensor", kwargs.get("tensor"))
        return tuple(getattr(obj, "shape", getattr(obj, "shape_metadata", ())))


@register_op("Frompyfunc")
class Frompyfunc(OpDef):
    """Wrap an arbitrary Python callable into an elementwise broadcasting operation."""

    op_name = "Frompyfunc"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer broadcasted output shape across input tensor operands.

        Args:
            *args (EagerTensorProtocol): Positional arguments (`func`, `nin`, `nout`, `*inputs`).
            **kwargs (TensorOrScalar): Keyword operation attributes.

        Returns:
            tuple[int, ...]: Broadcasted output shape tuple.
        """
        _ = kwargs
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        inps = args[3:] if len(args) > 3 else args
        shapes = [getattr(inp, "shape", getattr(inp, "shape_metadata", ())) for inp in inps if hasattr(inp, "shape") or hasattr(inp, "shape_metadata")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return tuple(res)


@register_op("Geomspace")
class Geomspace(OpDef):
    """Return numbers spaced evenly on a log scale (a geometric progression)."""

    op_name = "Geomspace"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer output shape for `Geomspace` based on `start`, `stop`, `num`, and `axis`.

        Args:
            *args (EagerTensorProtocol): Positional arguments (`start`, `stop`, `num`).
            **kwargs (TensorOrScalar): Keyword attributes (`num`, `axis`).

        Returns:
            tuple[int, ...]: Inferred output shape tuple.
        """
        start = args[0] if len(args) > 0 else kwargs.get("start")
        stop = args[1] if len(args) > 1 else kwargs.get("stop")
        raw_num = kwargs.get("num", args[2] if len(args) > 2 else 50)
        num: int = 50
        try:
            num = int(as_int(raw_num))
        except (TypeError, ValueError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        raw_axis = kwargs.get("axis", 0)
        axis: int = 0
        try:
            axis = int(as_int(raw_axis))
        except (TypeError, ValueError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        shape1 = getattr(start, "shape", getattr(start, "shape_metadata", ()))
        shape2 = getattr(stop, "shape", getattr(stop, "shape_metadata", ()))
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

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer output shape for `Geometric` from `size` or parameter `p`.

        Args:
            *args (EagerTensorProtocol): Positional arguments (`p`, `size`).
            **kwargs (TensorOrScalar): Keyword attributes (`p`, `size`).

        Returns:
            tuple[int, ...]: Inferred output shape tuple.
        """
        p = args[0] if len(args) > 0 else kwargs.get("p")
        size = kwargs.get("size", args[1] if len(args) > 1 else None)
        if size is None:
            return tuple(getattr(p, "shape", getattr(p, "shape_metadata", ())))
        if isinstance(size, int):
            return (size,)
        return tuple(int(as_int(d)) for d in as_eager_list(size))


def from_dlpack(obj: TensorOrScalar) -> TensorOrScalar:
    """Create a compiler tensor from an object implementing the DLPack protocol.

    Args:
        obj (TensorOrScalar): Input array or tensor supporting `__dlpack__`.

    Returns:
        TensorOrScalar: Imported compiler tensor.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("FromDlpack", obj)


def frompyfunc(func: TensorOrScalar, nin: int, nout: int) -> TensorOrScalar:
    """Wrap an arbitrary Python function into an elementwise universal function.

    Args:
        func (TensorOrScalar): Python callable to wrap.
        nin (int): Number of input arguments accepted by `func`.
        nout (int): Number of output values returned by `func`.

    Returns:
        TensorOrScalar: Wrapped universal function object.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Frompyfunc", func, nin, nout)


def geomspace(
    start: TensorOrScalar,
    stop: TensorOrScalar,
    num: int = 50,
    endpoint: bool = True,
    dtype: TensorOrScalar = None,
    axis: int = 0,
) -> TensorOrScalar:
    """Return numbers spaced evenly on a log scale (a geometric progression).

    Args:
        start (TensorOrScalar): Starting value of the sequence.
        stop (TensorOrScalar): Final value of the sequence.
        num (int): Number of samples to generate.
        endpoint (bool): Whether to include `stop` as the last sample.
        dtype (TensorOrScalar): Data type of the output tensor.
        axis (int): Axis in the result to store the samples.

    Returns:
        TensorOrScalar: Geometric progression tensor.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Geomspace", start, stop, num=num, endpoint=endpoint, dtype=dtype, axis=axis)


def geometric(p: TensorOrScalar, size: TensorOrScalar = None) -> TensorOrScalar:
    """Draw samples from the geometric distribution with success probability `p`.

    Args:
        p (TensorOrScalar): Probability of success on an individual trial.
        size (TensorOrScalar): Output shape tuple or integer length.

    Returns:
        TensorOrScalar: Tensor of samples drawn from the geometric distribution.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Geometric", p, size=size)
