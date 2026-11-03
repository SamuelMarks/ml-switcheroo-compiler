"""Implementation of builder operations and utilities for the tracing subsystem (`ml_switcheroo_compiler.tracing.builder`)."""

from __future__ import annotations

import typing
import uuid
from collections.abc import Sequence

from ml_switcheroo_ir import AttributeValue, LogicalNode

from ml_switcheroo_compiler.backends.eager_registry import (
    ShapeProtocol,
    as_callable,
    as_shape,
)
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    AttrValue,
    StructuralOperandProtocol,
    TensorOrScalar,
    is_backend_buffer,
    is_traced_data,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape_inference import infer_shape
from ml_switcheroo_compiler.ops.type_inference import resolve_output_dtype_and_device
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


@typing.runtime_checkable
class _ConstantCacheStateProtocol(typing.Protocol):
    """Protocol for tracing state objects exposing `constant_cache`."""

    constant_cache: dict[int, str]


def _to_attr_value(val: TensorOrScalar) -> AttributeValue:
    """Normalize a runtime value into an IR AttributeValue.

    Args:
        val (TensorOrScalar): Raw attribute value.

    Returns:
        AttributeValue: Normalized attribute value.
    """
    if isinstance(val, (int, float, str, bool, bytes)) or val is None:
        return val
    if isinstance(val, (list, tuple)):
        return [_to_attr_value(x) for x in val]
    if isinstance(val, dict):
        return {str(k): _to_attr_value(v) for k, v in val.items()}
    return str(val)


class TracingNodeBuilder:
    """Build tracing nodes."""

    @staticmethod
    def create_constant_node(val: TensorOrScalar, shape: Sequence[int | str]) -> str:
        """Create a constant node in the active tracing graph.

        Args:
            val (TensorOrScalar): The constant scalar or array value.
            shape (Sequence[int | str]): The shape of the constant tensor.

        Returns:
            str: The unique identifier of the created node.
        """
        out_id = str(uuid.uuid4())
        attrs: dict[str, AttributeValue] = {"value": _to_attr_value(val)}
        node = LogicalNode(
            id=out_id,
            op_type="Constant",
            attributes=attrs,
            shape_metadata=shape,
        )
        global_tracing_state.add_node(node)
        return out_id

    @staticmethod
    def extract_from_tensor(a: Tensor) -> tuple[str, tuple[int | str, ...]]:
        """Extract the node identifier and shape metadata from a tensor or proxy.

        Args:
            a (Tensor): The input tensor, proxy tensor, or raw array.

        Returns:
            tuple[str, tuple[int | str, ...]]: A tuple containing the node ID and its shape.
        """
        if is_traced_data(a.data):
            return str(a.data.id), a.shape
        data_id = id(a.data)
        raw_state: StructuralOperandProtocol = global_tracing_state
        if isinstance(raw_state, _ConstantCacheStateProtocol) and data_id in raw_state.constant_cache:
            return raw_state.constant_cache[data_id], a.shape
        val = a.data.tolist() if is_backend_buffer(a.data) else a.data
        out_id = TracingNodeBuilder.create_constant_node(val, a.shape)
        if isinstance(raw_state, _ConstantCacheStateProtocol):
            raw_state.constant_cache[data_id] = out_id
        return out_id, a.shape

    @staticmethod
    def extract_from_constant(a: TensorOrScalar | ProxyTensor) -> tuple[str | list[str], tuple[int | str, ...] | list[tuple[int | str, ...]]]:
        """Extract the node identifier and shape metadata from a constant value.

        Args:
            a (TensorOrScalar | ProxyTensor): The constant scalar, list, or array value.

        Returns:
            tuple[str | list[str], tuple[int | str, ...] | list[tuple[int | str, ...]]]: A tuple containing the node ID and its shape.
        """
        if isinstance(a, (list, tuple)) and any(type(x).__name__ in ("ProxyTensor", "Tensor") for x in a):
            ids: list[str] = []
            shapes: list[tuple[int | str, ...]] = []
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
        arr: TensorOrScalar = backend.array(a)
        val = arr.tolist() if is_backend_buffer(arr) else arr
        shape = as_shape(arr.shape) if isinstance(arr, ShapeProtocol) else ()
        out_id = TracingNodeBuilder.create_constant_node(val, shape)
        return out_id, shape

    @staticmethod
    def extract_proxy_inputs(
        args: Sequence[TensorOrScalar | ProxyTensor],
    ) -> tuple[list[str | list[str]], list[tuple[int | str, ...] | list[tuple[int | str, ...]]], Tensor | None]:
        """Extract proxy node IDs and shapes from a list of arguments.

        Args:
            args (Sequence[TensorOrScalar | ProxyTensor]): The positional arguments.

        Returns:
            tuple[list[str | list[str]], list[tuple[int | str, ...] | list[tuple[int | str, ...]]], Tensor | None]: Node IDs, shapes, and first tensor.
        """
        input_ids: list[str | list[str]] = []
        shapes: list[tuple[int | str, ...] | list[tuple[int | str, ...]]] = []
        first_tensor: Tensor | None = None

        for a in args:
            out_id: str | list[str]
            shape: tuple[int | str, ...] | list[tuple[int | str, ...]]
            if isinstance(a, Tensor):
                first_tensor = a if first_tensor is None else first_tensor
                out_id, shape = TracingNodeBuilder.extract_from_tensor(a)
            elif isinstance(a, ProxyTensor) or is_traced_data(a):
                raw_a: StructuralOperandProtocol = a
                out_id, shape = str(a.id), as_shape(raw_a.shape) if isinstance(raw_a, ShapeProtocol) else ()
            else:
                out_id, shape = TracingNodeBuilder.extract_from_constant(a)

            input_ids.append(out_id)
            shapes.append(shape)

        return input_ids, shapes, first_tensor

    @staticmethod
    def create_tracing_logical_node(
        op_type: str,
        input_ids: list[str],
        kwargs: dict[str, AttrValue],
        out_shape: Sequence[int | str],
    ) -> str:
        """Create and add a new logical node to the active tracing graph.

        Args:
            op_type (str): The type of the operation.
            input_ids (list[str]): The input node IDs.
            kwargs (dict[str, AttrValue]): The attributes for the node.
            out_shape (Sequence[int | str]): The output shape metadata.

        Returns:
            str: The unique identifier of the created node.
        """
        out_id = str(uuid.uuid4())
        attrs: dict[str, AttributeValue] = {str(k): _to_attr_value(v) for k, v in kwargs.items()}
        from ml_switcheroo_ir import SymInt, SymNode

        raw_inputs: Sequence[AttrValue] = input_ids
        flat_inputs: list[str] = [str(sub) for item in raw_inputs for sub in (item if isinstance(item, (list, tuple)) else (item,))]
        raw_out_shape: Sequence[AttrValue] = out_shape
        first_elem: AttrValue = raw_out_shape[0] if len(raw_out_shape) > 0 else None
        flat_shape: Sequence[AttrValue] = first_elem if isinstance(first_elem, (tuple, list)) else raw_out_shape
        norm_shape: tuple[int | str | SymNode | SymInt, ...] = tuple(int(d) if isinstance(d, float) else (d if isinstance(d, (int, str, SymNode, SymInt)) and not isinstance(d, bool) else ("?" if d is None else str(d))) for d in flat_shape)
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
    def emit_tracing_node(op_type: str, *args: TensorOrScalar | ProxyTensor, **kwargs: AttrValue) -> Tensor:
        """Emit a new tracing node and return its corresponding tensor.

        Args:
            op_type (str): The operation type.
            *args (TensorOrScalar | ProxyTensor): Positional arguments.
            **kwargs (AttrValue): Keyword arguments.

        Returns:
            Tensor: The newly created proxy tensor.
        """
        input_ids, shapes, first_tensor = TracingNodeBuilder.extract_proxy_inputs(args)

        inferred: TensorOrScalar = as_callable(infer_shape)(op_type, *shapes, **kwargs)
        out_shape: tuple[int | str, ...] = tuple(int(d) if isinstance(d, int) else str(d) for d in inferred) if isinstance(inferred, (tuple, list)) else ()
        out_dtype, device = resolve_output_dtype_and_device(first_tensor, kwargs)

        out_id = TracingNodeBuilder.create_tracing_logical_node(
            op_type,
            [str(i) for i in input_ids],
            kwargs,
            out_shape,
        )

        dtype_val = out_dtype.value if isinstance(out_dtype, DType) else str(out_dtype)
        proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=dtype_val)
        return Tensor(proxy, TensorConfig(out_shape, out_dtype, device))
