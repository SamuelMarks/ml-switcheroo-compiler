"""Creation node emission utilities and DLPack/ufunc/geometric creation operations."""

from __future__ import annotations

import logging
import typing
import uuid
from collections.abc import Sequence

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import (
    coerce_int,
    coerce_operand_list,
    get_active_backend,
)
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    OpValue,
    ShapeMetadataProtocol,
    ShapeProtocol,
    extract_shape,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import ensure_attrs
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state


@typing.runtime_checkable
class _ValueHolderProtocol(typing.Protocol):
    """Protocol for objects exposing a `.value` attribute."""

    value: OpValue


@typing.runtime_checkable
class _NameHolderProtocol(typing.Protocol):
    """Protocol for objects exposing a `.name` attribute."""

    name: OpValue


@typing.runtime_checkable
class _NdimProtocol(typing.Protocol):
    """Protocol for arrays exposing `.ndim`."""

    ndim: int


@typing.runtime_checkable
class _TolistProtocol(typing.Protocol):
    """Protocol for arrays exposing `.tolist()`."""

    def tolist(self) -> OpValue:
        """Convert array to nested Python list.

        Returns:
            OpValue: Nested Python list.
        """
        ...


@typing.runtime_checkable
class _ItemProtocol(typing.Protocol):
    """Protocol for arrays exposing `.item()`."""

    def item(self) -> OpValue:
        """Extract scalar Python value.

        Returns:
            OpValue: Scalar value.
        """
        ...


def _dtype_to_str(dtype: OpValue) -> str:
    """Resolve `dtype` to its string identifier.

    Args:
        dtype (OpValue): Data type object or string.

    Returns:
        str: String representation of `dtype`.
    """
    cand_dtype: OpValue | _ValueHolderProtocol = dtype
    dtype_name: OpValue | _NameHolderProtocol = dtype
    return str(
        cand_dtype.value
        if isinstance(cand_dtype, _ValueHolderProtocol)
        else (dtype_name.name if isinstance(dtype_name, _NameHolderProtocol) else dtype)
    )


def _emit_creation_node(
    op_type: str,
    shape: Sequence[int | str | OpValue],
    dtype: DType,
    attributes: OpValue = None,
) -> OpValue:
    """Emit a symbolic tensor creation node to the active tracing IR graph.

    Args:
        op_type (str): Canonical IR creation operation type name.
        shape (Sequence[int | str | OpValue]): Output tensor shape.
        dtype (DType): Output tensor data type.
        attributes (OpValue): Optional dictionary of node attributes.

    Returns:
        OpValue: Symbolic `Tensor` wrapping the emitted `LogicalNode`.

    Raises:
        RuntimeError: If called outside of an active tracing context.
    """
    from ml_switcheroo_compiler.ir.core import normalize_shape_metadata
    from ml_switcheroo_compiler.ir.shape_system import SymInt, SymNode

    if not global_tracing_state.is_tracing:
        msg = f"Cannot emit {op_type} node outside of a tracing context."
        raise RuntimeError(msg)

    norm_shape = tuple(d if isinstance(d, (int, str, SymNode, SymInt)) or d is None else int(str(d)) for d in shape)
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=[],
        attributes={**ensure_attrs(attributes or {}), "dtype": dtype.value},
        shape_metadata=normalize_shape_metadata(norm_shape),
    )
    global_tracing_state.add_node(node)

    proxy = ProxyTensor(
        id=out_id,
        shape=norm_shape,
        dtype=_dtype_to_str(dtype),
    )
    return Tensor(proxy, TensorConfig(norm_shape, dtype, config.default_device))


def _emit_constant_node(
    value: OpValue,
    dtype: DType,
) -> OpValue:
    """Emit a symbolic `Constant` node to the active tracing IR graph.

    Args:
        value (OpValue): Constant scalar or sequence value.
        dtype (DType): Output tensor data type.

    Returns:
        OpValue: Symbolic `Tensor` wrapping the emitted `Constant` node.

    Raises:
        RuntimeError: If called outside of an active tracing context.
    """
    if not global_tracing_state.is_tracing:
        msg = "Cannot emit Constant node outside of a tracing context."
        raise RuntimeError(msg)

    out_id = str(uuid.uuid4())
    dt_str = _dtype_to_str(dtype)
    val_arr = get_active_backend().array(
        value,
        dtype=dt_str,
    )
    raw_arr: OpValue = val_arr
    shape = extract_shape(raw_arr)
    ndim_val = raw_arr.ndim if isinstance(raw_arr, _NdimProtocol) else len(shape)
    cand_arr: OpValue | _TolistProtocol | _ItemProtocol = raw_arr
    val_attr: OpValue = (
        cand_arr.tolist()
        if ndim_val > 0 and isinstance(cand_arr, _TolistProtocol)
        else (cand_arr.item() if isinstance(cand_arr, _ItemProtocol) else val_arr)
    )

    node = LogicalNode(
        id=out_id,
        op_type="Constant",
        inputs=[],
        attributes=ensure_attrs({"value": val_attr}),
        shape_metadata=shape,
    )
    global_tracing_state.add_node(node)

    proxy = ProxyTensor(
        id=out_id,
        shape=shape,
        dtype=dt_str,
    )
    return Tensor(proxy, TensorConfig(shape, dtype, config.default_device))


@register_op("FromDlpack")
class FromDlpack(OpDef):
    """Create a compiler tensor from an external object implementing the DLPack protocol."""

    op_name = "FromDlpack"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> tuple[int, ...]:
        """Infer shape from the DLPack source tensor.

        Args:
            *args (OpValue): Positional DLPack source tensor arguments.
            **kwargs (OpValue): Keyword arguments (`ext_tensor` or `tensor`).

        Returns:
            tuple[int, ...]: Inferred tensor shape tuple.
        """
        obj = args[0] if len(args) > 0 else kwargs.get("ext_tensor", kwargs.get("tensor"))
        return extract_shape(obj)


@register_op("Frompyfunc")
class Frompyfunc(OpDef):
    """Wrap an arbitrary Python callable into an elementwise broadcasting operation."""

    op_name = "Frompyfunc"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> tuple[int, ...]:
        """Infer broadcasted output shape across input tensor operands.

        Args:
            *args (OpValue): Positional arguments (`func`, `nin`, `nout`, `*inputs`).
            **kwargs (OpValue): Keyword operation attributes.

        Returns:
            tuple[int, ...]: Broadcasted output shape tuple.
        """
        _ = kwargs
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        inps: Sequence[OpValue] = args[3:] if len(args) > 3 else args
        shapes = [extract_shape(inp) for inp in inps if isinstance(inp, (ShapeProtocol, ShapeMetadataProtocol))]
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

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> tuple[int, ...]:
        """Infer output shape for `Geomspace` based on `start`, `stop`, `num`, and `axis`.

        Args:
            *args (OpValue): Positional arguments (`start`, `stop`, `num`).
            **kwargs (OpValue): Keyword attributes (`num`, `axis`).

        Returns:
            tuple[int, ...]: Inferred output shape tuple.
        """
        start = args[0] if len(args) > 0 else kwargs.get("start")
        stop = args[1] if len(args) > 1 else kwargs.get("stop")
        raw_num = kwargs.get("num", args[2] if len(args) > 2 else 50)
        num: int = 50
        try:
            num = int(coerce_int(raw_num))
        except (TypeError, ValueError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        raw_axis = kwargs.get("axis", 0)
        axis: int = 0
        try:
            axis = int(coerce_int(raw_axis))
        except (TypeError, ValueError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        shape1 = extract_shape(start)
        shape2 = extract_shape(stop)
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

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> tuple[int, ...]:
        """Infer output shape for `Geometric` from `size` or parameter `p`.

        Args:
            *args (OpValue): Positional arguments (`p`, `size`).
            **kwargs (OpValue): Keyword attributes (`p`, `size`).

        Returns:
            tuple[int, ...]: Inferred output shape tuple.
        """
        p = args[0] if len(args) > 0 else kwargs.get("p")
        size = kwargs.get("size", args[1] if len(args) > 1 else None)
        if size is None:
            return extract_shape(p)
        if isinstance(size, int):
            return (size,)
        return tuple(int(coerce_int(d)) for d in coerce_operand_list(size))


def from_dlpack(obj: OpValue) -> OpValue:
    """Create a compiler tensor from an object implementing the DLPack protocol.

    Args:
        obj (OpValue): Input array or tensor supporting `__dlpack__`.

    Returns:
        OpValue: Imported compiler tensor.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("FromDlpack", obj)


def frompyfunc(func: OpValue, nin: int, nout: int) -> OpValue:
    """Wrap an arbitrary Python function into an elementwise universal function.

    Args:
        func (OpValue): Python callable to wrap.
        nin (int): Number of input arguments accepted by `func`.
        nout (int): Number of output values returned by `func`.

    Returns:
        OpValue: Wrapped universal function object.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Frompyfunc", func, nin, nout)


def geomspace(
    start: OpValue,
    stop: OpValue,
    num: int = 50,
    endpoint: bool = True,
    dtype: OpValue = None,
    axis: int = 0,
) -> OpValue:
    """Return numbers spaced evenly on a log scale (a geometric progression).

    Args:
        start (OpValue): Starting value of the sequence.
        stop (OpValue): Final value of the sequence.
        num (int): Number of samples to generate.
        endpoint (bool): Whether to include `stop` as the last sample.
        dtype (OpValue): Data type of the output tensor.
        axis (int): Axis in the result to store the samples.

    Returns:
        OpValue: Geometric progression tensor.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Geomspace", start, stop, num=num, endpoint=endpoint, dtype=dtype, axis=axis)


def geometric(p: OpValue, size: OpValue = None) -> OpValue:
    """Draw samples from the geometric distribution with success probability `p`.

    Args:
        p (OpValue): Probability of success on an individual trial.
        size (OpValue): Output shape tuple or integer length.

    Returns:
        OpValue: Tensor of samples drawn from the geometric distribution.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Geometric", p, size=size)
