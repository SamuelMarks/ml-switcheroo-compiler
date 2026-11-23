"""Operations and utilities for `ml_switcheroo_compiler.tracing.builder`."""

from __future__ import annotations

import typing
import uuid
from collections.abc import Mapping, Sequence

from ml_switcheroo_ir import AttributeValue, LogicalNode

from ml_switcheroo_compiler.backends.eager_registry import ShapeProtocol
from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    IRAttrValue,
    OpValue,
    ShapeDim,
    ShapeTuple,
    is_backend_buffer,
    is_traced_data,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape_inference import infer_shape
from ml_switcheroo_compiler.ops.type_inference import resolve_output_dtype_and_device
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


def _has_builder_shape(val: OpValue) -> typing.TypeGuard[ShapeProtocol]:
    """Check whether `val` exposes `.shape`.

    Args:
        val (OpValue): Candidate proxy object.

    Returns:
        typing.TypeGuard[ShapeProtocol]: `True` when `.shape` is available.
    """
    found: bool = isinstance(val, ShapeProtocol)
    return found or is_mock_with_attrs(val, "shape")


def _has_readable_tensor_data(val: Tensor) -> bool:
    """Check whether `val` exposes a readable `.data` attribute without raising `AttributeError`.

    Args:
        val (Tensor): Candidate tensor object.

    Returns:
        bool: `True` when `.data` is present and readable.
    """
    try:
        _ = val.data
    except AttributeError:
        return False
    return True


def _coerce_attr_val(val: OpValue | IRAttrValue) -> AttributeValue:
    """Coerce a runtime value to an `AttributeValue` for IR node storage.

    Args:
        val (OpValue | IRAttrValue): Runtime attribute value.

    Returns:
        AttributeValue: IR-compatible attribute value.
    """
    if isinstance(val, (int, float, str, bool, bytes)) or val is None:
        return val
    if isinstance(val, (list, tuple)):
        return [_coerce_attr_val(v) for v in val]
    return str(val)


def _extract_builder_shape(seq: Sequence[ShapeDim]) -> tuple[int, ...]:
    """Convert a sequence of dimensions into a concrete integer shape tuple.

    Args:
        seq (Sequence[ShapeDim]): Shape sequence.

    Returns:
        tuple[int, ...]: Concrete integer shape tuple.
    """
    return tuple(int(d) if isinstance(d, int) else 0 for d in seq)


@typing.runtime_checkable
class _ConstantCacheStateProtocol(typing.Protocol):
    """Protocol for tracing state objects exposing a `constant_cache` dictionary."""

    constant_cache: dict[int, str]


class TracingNodeBuilder:
    """Build tracing nodes."""

    @staticmethod
    def create_constant_node(val: OpValue, shape: Sequence[ShapeDim]) -> str:
        """Create a constant node in the active tracing graph.

        Args:
            val (OpValue): The constant scalar or array value.
            shape (Sequence[ShapeDim]): The shape of the constant tensor.

        Returns:
            str: The unique identifier of the created node.
        """
        out_id = str(uuid.uuid4())
        attrs: dict[str, AttributeValue] = {"value": _coerce_attr_val(val)}
        from ml_switcheroo_compiler.ir.core import normalize_shape_metadata

        node = LogicalNode(
            id=out_id,
            op_type="Constant",
            attributes=attrs,
            shape_metadata=normalize_shape_metadata(shape),
        )
        global_tracing_state.add_node(node)
        return out_id

    @staticmethod
    def extract_from_tensor(a: Tensor) -> tuple[str, ShapeTuple]:
        """Extract the node identifier and shape metadata from a tensor or proxy.

        Args:
            a (Tensor): The input tensor, proxy tensor, or raw array.

        Returns:
            tuple[str, ShapeTuple]: A tuple containing the node ID and its shape.
        """
        if is_traced_data(a.data):
            return str(a.data.id), a.shape
        data_id = id(a.data)
        if data_id in global_tracing_state.constant_cache:
            return str(global_tracing_state.constant_cache[data_id]), a.shape
        val = a.data.tolist() if is_backend_buffer(a.data) else a.data
        out_id = TracingNodeBuilder.create_constant_node(val, a.shape)
        global_tracing_state.constant_cache[data_id] = out_id
        return out_id, a.shape

    @staticmethod
    def extract_from_constant(a: OpValue | ProxyTensor) -> tuple[str | list[str], ShapeTuple | list[ShapeTuple]]:
        """Extract the node identifier and shape metadata from a constant value.

        Args:
            a (OpValue | ProxyTensor): The constant scalar, list, or array value.

        Returns:
            tuple[str | list[str], ShapeTuple | list[ShapeTuple]]: A tuple containing the node ID and its shape.
        """
        if isinstance(a, (list, tuple)) and any(type(x).__name__ in ("ProxyTensor", "Tensor") for x in a):
            ids: list[str] = []
            shapes: list[ShapeTuple] = []
            for x in a:
                if isinstance(x, ProxyTensor) or is_traced_data(x):
                    ids.append(str(x.id))
                    shapes.append(tuple(x.shape))
                else:
                    oid, oshape = TracingNodeBuilder.extract_from_constant(x)
                    ids.append(oid if isinstance(oid, str) else str(oid[0]))
                    shapes.append(oshape if isinstance(oshape, tuple) else tuple(oshape[0]))
            return ids, shapes

        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        arr: OpValue = backend.array(a)
        val = arr.tolist() if is_backend_buffer(arr) else arr
        shape = _extract_builder_shape(arr.shape) if isinstance(arr, ShapeProtocol) else ()
        out_id = TracingNodeBuilder.create_constant_node(val, shape)
        return out_id, shape

    @staticmethod
    def extract_proxy_inputs(
        args: Sequence[OpValue | ProxyTensor],
    ) -> tuple[list[str | list[str]], list[ShapeTuple | list[ShapeTuple]], Tensor | None]:
        """Extract proxy node IDs and shapes from a list of arguments.

        Args:
            args (Sequence[OpValue | ProxyTensor]): The positional arguments.

        Returns:
            tuple[list[str | list[str]], list[ShapeTuple | list[ShapeTuple]], Tensor | None]: Node IDs, shapes, and
                first tensor.
        """
        input_ids: list[str | list[str]] = []
        shapes: list[ShapeTuple | list[ShapeTuple]] = []
        first_tensor: Tensor | None = None

        for a in args:
            out_id: str | list[str]
            shape: ShapeTuple | list[ShapeTuple]
            if isinstance(a, Tensor) and _has_readable_tensor_data(a):
                first_tensor = a if first_tensor is None else first_tensor
                out_id, shape = TracingNodeBuilder.extract_from_tensor(a)
            elif isinstance(a, ProxyTensor) or is_traced_data(a):
                raw_a: OpValue = a
                out_id, shape = (
                    str(a.id),
                    (_extract_builder_shape(raw_a.shape) if _has_builder_shape(raw_a) else ()),
                )
            elif isinstance(a, Tensor):
                first_tensor = a if first_tensor is None else first_tensor
                out_id = TracingNodeBuilder.create_constant_node(0, a.shape)
                shape = a.shape
            else:
                out_id, shape = TracingNodeBuilder.extract_from_constant(a)

            input_ids.append(out_id)
            shapes.append(shape)

        return input_ids, shapes, first_tensor

    @staticmethod
    def create_tracing_logical_node(
        op_type: str,
        input_ids: list[str],
        kwargs: Mapping[str, OpValue],
        out_shape: Sequence[ShapeDim],
        node_id: str | None = None,
    ) -> str:
        """Create and add a new logical node to the active tracing graph.

        Args:
            op_type (str): The type of the operation.
            input_ids (list[str]): The input node IDs.
            kwargs (Mapping[str, OpValue]): The attributes for the node.
            out_shape (Sequence[ShapeDim]): The output shape metadata.
            node_id (str | None): Optional explicit node identifier; defaults to a generated UUID.

        Returns:
            str: The unique identifier of the created node.
        """
        out_id = node_id if node_id is not None else str(uuid.uuid4())
        attrs: dict[str, AttributeValue] = {str(k): _coerce_attr_val(v) for k, v in kwargs.items()}
        from ml_switcheroo_compiler.ir import SymInt, SymNode

        raw_inputs: Sequence[IRAttrValue] = input_ids
        flat_inputs: list[str] = [
            str(sub) for item in raw_inputs for sub in (item if isinstance(item, (list, tuple)) else (item,))
        ]
        raw_out_shape: Sequence[IRAttrValue] = out_shape
        first_elem: IRAttrValue = raw_out_shape[0] if len(raw_out_shape) > 0 else None
        flat_shape: Sequence[IRAttrValue] = first_elem if isinstance(first_elem, (tuple, list)) else raw_out_shape
        norm_shape: tuple[int | str | SymNode | SymInt, ...] = tuple(
            int(d)
            if isinstance(d, float)
            else (
                d
                if isinstance(d, (int, str, SymNode, SymInt)) and not isinstance(d, bool)
                else ("?" if d is None else str(d))
            )
            for d in flat_shape
        )
        node = LogicalNode(
            id=out_id,
            op_type=op_type,
            inputs=flat_inputs,
            attributes=attrs,
            shape_metadata=norm_shape,
        )
        global_tracing_state.add_node(node)
        return out_id

    @staticmethod
    def emit_tracing_node(op_type: str, *args: OpValue | ProxyTensor, **kwargs: OpValue) -> Tensor:
        """Emit a new tracing node and return its corresponding tensor.

        Args:
            op_type (str): The operation type.
            *args (OpValue | ProxyTensor): Positional arguments.
            **kwargs (OpValue): Keyword arguments.

        Returns:
            Tensor: The newly created proxy tensor.
        """
        input_ids, shapes, first_tensor = TracingNodeBuilder.extract_proxy_inputs(args)

        norm_shapes: list[tuple[int, ...] | None] = [
            _extract_builder_shape(s) if isinstance(s, tuple) else None for s in shapes
        ]
        norm_kwargs: dict[str, int | float | str | bool | tuple[int, ...] | Sequence[int] | None] = {
            k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool, tuple, list)) or v is None
        }
        inferred = infer_shape(op_type, *norm_shapes, **norm_kwargs)
        out_shape: tuple[int | str, ...] = tuple(int(d) if isinstance(d, int) else str(d) for d in (inferred or ()))
        out_dtype, device = resolve_output_dtype_and_device(first_tensor, dict(kwargs))

        out_id = TracingNodeBuilder.create_tracing_logical_node(
            op_type,
            [str(i) for i in input_ids],
            kwargs,
            out_shape,
        )

        dtype_val = out_dtype.value if isinstance(out_dtype, DType) else str(out_dtype)
        proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=dtype_val)
        return Tensor(proxy, TensorConfig(out_shape, out_dtype, device))
